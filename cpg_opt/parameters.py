import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "optimization"))

from core.parameters import ParameterSpec, ParameterSpace


def get_cpg_parameter_space() -> ParameterSpace:
    specs = [
        ParameterSpec("stdp_lambda", 0.0001, 0.01, 0.001, log_scale=True),
        ParameterSpec("stdp_alpha", 0.5, 1.5, 0.95),
        ParameterSpec("stdp_mu_plus", 0.0, 1.0, 0.4),
        ParameterSpec("stdp_mu_minus", 0.0, 1.0, 0.4),
        ParameterSpec("stdp_wmax", 40.0, 250.0, 120.0),
        ParameterSpec("w0_in", 5.0, 60.0, 22.0),

        ParameterSpec("bs_rate_amp_hz", 20.0, 400.0, 80.0),
        ParameterSpec("bs_osc_hz", 0.3, 3.0, 1.0),
        ParameterSpec("base_drive_hz", 0.5, 15.0, 2.0),
        ParameterSpec("base_drive_w", 0.3, 8.0, 1.0),
        ParameterSpec("cut_rate_on_hz", 50.0, 500.0, 200.0),

        ParameterSpec("w_rg_recip", -30.0, -5.0, -18.0),
        ParameterSpec("w_comm_inh", -25.0, -2.0, -10.0),
        ParameterSpec("p_comm", 0.02, 0.20, 0.08),
        ParameterSpec("ia2rg_w", 2.0, 30.0, 12.0),
        ParameterSpec("w0_rm", 10.0, 60.0, 30.0),

        ParameterSpec("stretch_gain", 0.05, 1.0, 0.35),
        ParameterSpec("ia_k_force", 1.0, 20.0, 6.0),
    ]
    return ParameterSpace(specs=specs)


def get_reduced_cpg_parameter_space() -> ParameterSpace:
    specs = [
        ParameterSpec("stdp_lambda", 0.0001, 0.01, 0.001, log_scale=True),
        ParameterSpec("stdp_alpha", 0.5, 1.5, 0.95),
        ParameterSpec("stdp_mu_plus", 0.0, 1.0, 0.4),
        ParameterSpec("stdp_mu_minus", 0.0, 1.0, 0.4),
        ParameterSpec("stdp_wmax", 40.0, 250.0, 120.0),
        ParameterSpec("w0_in", 5.0, 60.0, 22.0),

        ParameterSpec("bs_rate_amp_hz", 20.0, 400.0, 80.0),
        ParameterSpec("bs_osc_hz", 0.3, 3.0, 1.0),
        ParameterSpec("base_drive_hz", 0.5, 15.0, 2.0),
        ParameterSpec("base_drive_w", 0.3, 8.0, 1.0),
        ParameterSpec("cut_rate_on_hz", 50.0, 500.0, 200.0),
    ]
    return ParameterSpace(specs=specs)
