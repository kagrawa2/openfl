# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import logging

log = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command line arguments to provide the required parameters for running the tests.

    Returns:
        argparse.Namespace: Parsed command line arguments with the following attributes:
            - results_dir (str, optional): Directory to store the results
            - num_collaborators (int, default=2): Number of collaborators
            - num_rounds (int, default=5): Number of rounds to train
            - model_name (str, default="torch/mnist"): Model name
            - disable_client_auth (bool): Disable client authentication
            - disable_tls (bool): Disable TLS for communication
            - log_memory_usage (bool): Enable Memory leak logs

    Raises:
        SystemExit: If the required arguments are not provided or if any argument parsing error occurs.
    """
    try:
        parser = argparse.ArgumentParser(description="Provide the required arguments to run the tests")
        parser.add_argument("--num_collaborators", type=int, default=2, help="Number of collaborators. Default is 2")
        parser.add_argument("--num_rounds", type=int, default=5, help="Number of rounds to train. Default is 5")
        parser.add_argument("--model_name", type=str, help="Model name. Not required for Workflow APIs")
        parser.add_argument("--disable_client_auth", action="store_true", help="Disable client authentication. Default is False")
        parser.add_argument("--disable_tls", action="store_true", help="Disable TLS for communication. Default is False")
        parser.add_argument("--log_memory_usage", action="store_true", help="Enable Memory leak logs. Default is False")
        args = parser.parse_known_args()[0]
        return args

    except Exception as e:
        log.error(f"Failed to parse arguments: {e}")
        sys.exit(1)
