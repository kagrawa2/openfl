# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import shutil
import xml.etree.ElementTree as ET
import logging
from pathlib import Path

from tests.end_to_end.utils.logger import configure_logging
from tests.end_to_end.utils.logger import logger as log
from tests.end_to_end.utils.conftest_helper import parse_arguments
import tests.end_to_end.utils.docker_helper as dh


def pytest_addoption(parser):
    """
    Add custom command line options to the pytest parser.
    Args:
        parser: pytest parser object
    """
    parser.addini("results_dir", "Directory to store test results", default="results")
    parser.addini("log_level", "Logging level", default="DEBUG")
    parser.addoption("--num_collaborators")
    parser.addoption("--num_rounds")
    parser.addoption("--model_name")
    parser.addoption("--disable_client_auth", action="store_true")
    parser.addoption("--disable_tls", action="store_true")
    parser.addoption("--log_memory_usage", action="store_true")


def pytest_configure(config):
    """
    Configure the pytest plugin.
    Args:
        config: pytest config object
    """
    # Declare some global variables
    args = parse_arguments()
    # Use the model name from the test case name if not provided as a command line argument
    config.model_name = args.model_name
    config.num_collaborators = args.num_collaborators
    config.num_rounds = args.num_rounds
    config.require_client_auth = not args.disable_client_auth
    config.use_tls = not args.disable_tls
    config.log_memory_usage = args.log_memory_usage
    config.results_dir = config.getini("results_dir")


@pytest.fixture(scope="session", autouse=True)
def setup_logging(pytestconfig):
    """
    Setup logging for the test session.
    Args:
        pytestconfig: pytest config object
    Returns:
        logger: logger object
    """
    tmp_results_dir = pytestconfig.getini("results_dir")
    log_level = pytestconfig.getini("log_level")

    results_dir = os.path.join(Path().home(), tmp_results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Setup a global logger to ensure logging works before any test-specific logs are set
    configure_logging(f"{results_dir}/deployment.log", log_level)
    return logging.getLogger()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to capture the result of setup, call, and teardown phases.
    This avoids duplicate entries for Pass/Fail in the XML report.
    """
    outcome = yield
    report = outcome.get_result()

    # Retrieve the custom test_id marker if it exists
    test_id_marker = item.get_closest_marker("test_id")
    outcome_mapping = {"passed": "Pass", "failed": "Fail"}
    report_when_mapping = {"setup": "Setup", "call": "Test", "teardown": "Teardown"}
    final_outcome = outcome_mapping.get(report.outcome, report.outcome)
    report_phase = report_when_mapping.get(report.when, report.when)

    # Modify nodeid if test_id is provided and append outcome and phase
    if test_id_marker:
        test_id = test_id_marker.args[0]
        report.nodeid = (
            f"{report.nodeid} [{test_id}] [outcome: {final_outcome}] [phase: {report_phase}]"
        )

    # Initialize XML structure if not already initialized
    if not hasattr(item.config, "_xml_report"):
        item.config._xml_report = ET.Element(
            "testsuite",
            {
                "name": "pytest",
                "errors": "0",
                "failures": "0",
                "skipped": "0",
                "tests": "0",
                "time": "0",
                "timestamp": "",
                "hostname": "",
            },
        )

    # Store the result of each phase (setup/call/teardown)
    if not hasattr(item, "_results"):
        item._results = {}

    # Save the outcome and other details per phase
    item._results[report.when] = {
        "outcome": final_outcome,
        "longrepr": report.longrepr,
        "duration": report.duration,
    }
    # Log failures
    if report.when == "call" and report.failed:
        logger = logging.getLogger()
        logger.error(f"Test {report.nodeid} failed: {call.excinfo.value}")

    # Only create the XML element after the teardown phase
    if report.when == "teardown" and not hasattr(item, "_xml_created"):
        item._xml_created = True  # Ensure XML creation happens only once

        # Determine final outcome based on the worst phase result
        if "call" in item._results:
            final_outcome = item._results["call"]["outcome"]
        elif "setup" in item._results:
            final_outcome = item._results["setup"]["outcome"]
        else:
            final_outcome = "skipped"

        # Create the <testcase> XML element
        testcase = ET.SubElement(
            item.config._xml_report,
            "testcase",
            {
                "classname": item.module.__name__,
                "name": item.name,
                "time": str(sum(result["duration"] for result in item._results.values())),
            },
        )

        # Add <failure> or <skipped> tags based on the final outcome
        if final_outcome == "Fail":
            failure_message = item._results.get("call", {}).get(
                "longrepr", item._results.get("setup", {}).get("longrepr", "Unknown Error")
            )
            failure = ET.SubElement(
                testcase,
                "error",
                {
                    "message": str(failure_message),
                },
            )
            failure.text = str(failure_message)
        elif final_outcome == "skipped":
            skipped_message = item._results.get("setup", {}).get("longrepr", "Skipped")
            skipped = ET.SubElement(
                testcase,
                "skipped",
                {
                    "message": str(skipped_message),
                },
            )
            skipped.text = str(skipped_message)

        # Update the testsuite summary statistics
        tests = int(item.config._xml_report.attrib["tests"]) + 1
        item.config._xml_report.attrib["tests"] = str(tests)
        if final_outcome == "Fail":
            failures = int(item.config._xml_report.attrib["failures"]) + 1
            item.config._xml_report.attrib["failures"] = str(failures)
        elif final_outcome == "skipped":
            skipped = int(item.config._xml_report.attrib["skipped"]) + 1
            item.config._xml_report.attrib["skipped"] = str(skipped)


def pytest_sessionfinish(session, exitstatus):
    """
    Operations to be performed after the test session is finished.
    More functionalities to be added in this function in future.
    """
    cache_dir = os.path.join(session.config.rootdir, ".pytest_cache")
    log.debug(f"\nClearing .pytest_cache directory at {cache_dir}")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=False)
        log.debug(f"Cleared .pytest_cache directory at {cache_dir}")

    # Cleanup docker containers related to aggregator and collaborators, if any.
    dh.cleanup_docker_containers(list_of_containers=["aggregator", "collaborator*"])
    # Cleanup docker network created for openfl, if any.
    dh.remove_docker_network(["openfl"])
