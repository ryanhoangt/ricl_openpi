import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Checkpoint:
    """Load a traj perceiver policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_fast_libero_traj_perceiver").
    config: str
    # Checkpoint directory.
    dir: str
    # Directory containing demo subdirectories (each with processed_demo.npz).
    # The first demo is used as the fixed reference trajectory.
    demos_dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the traj perceiver policy server."""

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    policy: Checkpoint = dataclasses.field(default_factory=lambda: Checkpoint("", "", ""))


def main(args: Args) -> None:
    policy = _policy_config.create_trained_traj_perceiver_policy(
        _config.get_config(args.policy.config),
        args.policy.dir,
        demos_dir=args.policy.demos_dir,
    )

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
