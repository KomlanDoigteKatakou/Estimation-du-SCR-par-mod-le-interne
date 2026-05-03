#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <chrono>
#include <iomanip>

using namespace std;

// ─────────────────────────────────────────────
//  Parameters
// ─────────────────────────────────────────────
struct Parameters {
    // Asset / rate model (calibrated on German data 1998-2008)
    double mu     = 0.0425;
    double sigmaA = 0.0428;
    double kappa  = 0.1449;
    double xi     = 0.0364;   // long-run mean under P
    double sigmaR = 0.006;
    double r0     = 0.0419;
    double rho    = -0.0597;
    double lambda = -0.5061;  // market price of risk

    // Contract
    double g     = 0.035;
    double delta = 0.9;
    double y     = 0.5;
    double L0    = 10000.0;
    double x0    = 0.1;
    int    T     = 10;

    // Derived
    double R0, A0, xi_tilde;

    // Simulation budget (from Bauer et al. 2010, Table 2)
    int K0 = 1500000;
    int N  = 320000;
    int K1 = 300;

    double alpha = 0.995;

    Parameters() {
        R0       = x0 * L0;
        A0       = L0 + R0;
        xi_tilde = xi - (lambda * sigmaR) / kappa;  // risk-neutral long-run mean
    }
};

// ─────────────────────────────────────────────
//  Random number generator
// ─────────────────────────────────────────────
class RNG {
    mt19937_64 gen;
    normal_distribution<double> nd{0.0, 1.0};
public:
    explicit RNG(uint64_t seed = 12345) : gen(seed) {}
    double normal() { return nd(gen); }
    void seed(uint64_t s) { gen.seed(s); }
};

// ─────────────────────────────────────────────
//  Vasicek exact simulation
//
//  Returns  r_{t+dt}  and  I = ∫_t^{t+dt} r_u du  (both exact).
//  Also returns dW_r (the Brownian increment of the rate), which must
//  be used as the correlated component of the asset Brownian motion.
//
//  Joint distribution of (∫r du, W_T - W_t):
//    [I ]   ~  N( mean_I, [[var_I,  cov_IW],
//    [dW]                   [cov_IW, dt    ]] )
//  solved via Cholesky.
// ─────────────────────────────────────────────
struct VasicekOut { double r_new, I, dW; };

static VasicekOut vasicek_exact(double r, double dt,
                                double kappa, double xi_level,
                                double sigmaR, RNG& rng) {
    double ek  = exp(-kappa * dt);
    double e2k = exp(-2.0 * kappa * dt);

    // Variances / covariance of (dW, K) where K = ∫_0^dt (r_s - r_0*e^{-ks}) ds
    // We use the parameterisation:
    //   dW ~ N(0, dt)
    //   K  ~ N(0, (1-e^{-2kappa*dt})/(2*kappa))
    //   Cov(dW,K) = (1 - e^{-kappa*dt}) / kappa
    double var_dW  = dt;
    double var_K   = (1.0 - e2k) / (2.0 * kappa);
    double cov_dWK = (1.0 - ek) / kappa;

    // Cholesky: L = [[l11,0],[l21,l22]]
    double l11 = sqrt(var_dW);
    double l21 = cov_dWK / l11;
    double l22 = sqrt(max(var_K - l21 * l21, 0.0));

    double z1 = rng.normal();
    double z2 = rng.normal();
    double dW = l11 * z1;
    double K  = l21 * z1 + l22 * z2;

    // r_{t+dt}
    double r_new = ek * r + xi_level * (1.0 - ek) + sigmaR * K;

    // ∫_t^{t+dt} r_u du  (exact, uses K and dW)
    //   I = xi*dt + (r - xi)*(1-ek)/kappa + sigmaR/kappa * (dW - K)
    double I = xi_level * dt
             + (r - xi_level) * (1.0 - ek) / kappa
             + (sigmaR / kappa) * (dW - K);

    return {r_new, I, dW};
}

// ─────────────────────────────────────────────
//  Asset exact simulation
//
//  Under P:   drift term = mu * dt
//  Under Q:   drift term = I  (= ∫r du, replaces mu*dt)
//
//  dW_r is already drawn; we draw one independent normal for the
//  orthogonal component dZ ~ N(0,dt).
// ─────────────────────────────────────────────
static double asset_exact(double A, double dt, double sigmaA, double rho,
                           double dW_r, double drift, RNG& rng) {
    // dZ_perp is the component of the asset BM orthogonal to dW_r
    double dZ = rng.normal() * sqrt(dt);
    double bm  = sigmaA * (rho * dW_r + sqrt(1.0 - rho * rho) * dZ);
    return A * exp(drift - 0.5 * sigmaA * sigmaA * dt + bm);
}

// ─────────────────────────────────────────────
//  Contract update (MUST bonus rule)
// ─────────────────────────────────────────────
struct ContractOut { double L, d, c, A_plus, R; };

static ContractOut contract_update(double A_minus, double A_plus_prev,
                                   double L_prev, double g, double delta, double y) {
    double earnings  = A_minus - A_plus_prev;
    double book_earn = y * earnings;
    double guaranteed = (1.0 + g) * L_prev;

    double bonus = max(delta * book_earn - g * L_prev, 0.0);
    double L_new = guaranteed + bonus;

    double d = 0.0;
    if      (delta * book_earn > g * L_prev) d = (1.0 - delta) * book_earn;
    else if (book_earn         > g * L_prev) d = book_earn - g * L_prev;

    double c = max(L_new - A_minus, 0.0);
    if (c > 0.0) d = 0.0;

    double A_plus_new = A_minus - d + c;
    double R_new      = A_plus_new - L_new;

    return {L_new, d, c, A_plus_new, R_new};
}

// ─────────────────────────────────────────────
//  V0 = E^Q[ sum_t B_{0,t}^{-1} * cashflow_t ]
//  We accumulate the discount using the exact integral I.
//  The only terminal cashflow considered is L_T (Bauer 2010 setup:
//  policyholders receive L_T at maturity, shareholders receive
//  dividends net of capital contributions along the way).
//  AC0 = A0 - E^Q[B_{0,T}^{-1} * L_T]
// ─────────────────────────────────────────────
static double estimate_AC0(const Parameters& p, int K0, RNG& rng) {
    double sum = 0.0;
    for (int k = 0; k < K0; ++k) {
        double A = p.A0, r = p.r0, L = p.L0, disc = 1.0;
        for (int t = 1; t <= p.T; ++t) {
            auto vout = vasicek_exact(r, 1.0, p.kappa, p.xi_tilde, p.sigmaR, rng);
            // FIX: use exact discount factor exp(-I)
            disc *= exp(-vout.I);
            double A_minus = asset_exact(A, 1.0, p.sigmaA, p.rho, vout.dW, vout.I, rng);
            auto cout_ = contract_update(A_minus, A, L, p.g, p.delta, p.y);
            A = cout_.A_plus;
            L = cout_.L;
            r = vout.r_new;
        }
        sum += disc * L;  // B_{0,T}^{-1} * L_T
    }
    return p.A0 - sum / static_cast<double>(K0);
}

// ─────────────────────────────────────────────
//  V1(state) = E^Q[ B_{1,T}^{-1} * L_T | state at t=1 ]
//  Returns the estimated V1 using K1 inner paths.
// ─────────────────────────────────────────────
static double estimate_V1(double A1, double r1, double L1,
                           const Parameters& p, int K1, RNG& rng) {
    double sum = 0.0;
    for (int k = 0; k < K1; ++k) {
        double A = A1, r = r1, L = L1, disc = 1.0;
        for (int t = 2; t <= p.T; ++t) {
            auto vout = vasicek_exact(r, 1.0, p.kappa, p.xi_tilde, p.sigmaR, rng);
            // FIX: exact discount factor
            disc *= exp(-vout.I);
            double A_minus = asset_exact(A, 1.0, p.sigmaA, p.rho, vout.dW, vout.I, rng);
            auto cout_ = contract_update(A_minus, A, L, p.g, p.delta, p.y);
            A = cout_.A_plus;
            L = cout_.L;
            r = vout.r_new;
        }
        sum += disc * L;
    }
    return A1 - sum / static_cast<double>(K1);
}

// ─────────────────────────────────────────────
//  Main SCR estimation via nested simulations
// ─────────────────────────────────────────────
double nested_SCR(const Parameters& p) {
    RNG rng(123456789);

    // Step 1 – estimate AC0
    cout << "Estimating AC0 with K0=" << p.K0 << "..." << endl;
    double AC0 = estimate_AC0(p, p.K0, rng);
    cout << "  AC0 = " << AC0 << endl;

    // Step 2 – generate N outer (P-measure) scenarios for year 1
    cout << "Generating " << p.N << " P-scenarios..." << endl;
    struct Scenario { double A1, r1, L1, X1; };
    vector<Scenario> scenarios(p.N);

    for (int i = 0; i < p.N; ++i) {
        double A = p.A0, r = p.r0, L = p.L0;

        // FIX: under P, drift = mu*dt, and dW_r enters the asset BM correctly
        // The rate BM under P uses xi (not xi_tilde)
        auto vout = vasicek_exact(r, 1.0, p.kappa, p.xi, p.sigmaR, rng);
        double A_minus = asset_exact(A, 1.0, p.sigmaA, p.rho, vout.dW, p.mu * 1.0, rng);
        auto cout_ = contract_update(A_minus, A, L, p.g, p.delta, p.y);

        scenarios[i] = {
            cout_.A_plus,
            vout.r_new,
            cout_.L,
            cout_.d - cout_.c   // X1 = dividend - capital contribution
        };
    }

    // Step 3 – inner simulations under Q for each outer scenario
    cout << "Running inner simulations (K1=" << p.K1 << ")..." << endl;
    vector<double> losses(p.N);
    double i_rate = p.r0;   // risk-free rate used to discount AC1 back to t=0

    for (int i = 0; i < p.N; ++i) {
        const auto& s = scenarios[i];
        double V1  = estimate_V1(s.A1, s.r1, s.L1, p, p.K1, rng);
        double AC1 = V1 + s.X1;
        losses[i]  = AC0 - AC1 / (1.0 + i_rate);
    }

    // Step 4 – quantile at level alpha
    // FIX: correct order statistic — index = ceil(alpha * N) - 1 (0-based)
    sort(losses.begin(), losses.end());
    int idx = static_cast<int>(ceil(p.alpha * static_cast<double>(p.N))) - 1;
    idx = max(0, min(idx, p.N - 1));

    return losses[idx];
}

// ─────────────────────────────────────────────
int main() {
    cout << fixed << setprecision(4);
    Parameters p;

    cout << "=== Nested Simulations — SCR estimation ===" << endl;
    cout << "K0=" << p.K0 << "  N=" << p.N << "  K1=" << p.K1 << endl;

    auto t0 = chrono::high_resolution_clock::now();
    double scr = nested_SCR(p);
    auto t1 = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(t1 - t0).count();

    cout << "\nSCR = " << scr << endl;
    cout << "Time = " << elapsed << " s" << endl;
    cout << "Reference (Bauer 2010 Table 2): ~1249.7" << endl;

    return 0;
}
