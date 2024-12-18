# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Workspace module."""

import logging
import os
import shutil
import subprocess  # nosec
import sys
import tempfile
from hashlib import sha256
from pathlib import Path
from subprocess import check_call  # nosec
from sys import executable
from typing import Union

from click import Choice, echo, group, option, pass_context
from click import Path as ClickPath
from cryptography.hazmat.primitives import serialization

from openfl.cryptography.ca import generate_root_cert, generate_signing_csr, sign_certificate
from openfl.federated.plan import Plan
from openfl.interface import plan
from openfl.interface.cli_helper import CERT_DIR, OPENFL_USERDIR, SITEPACKS, WORKSPACE, print_tree


@group()
@pass_context
def workspace(context):
    """Manage Federated Learning Workspaces.

    Args:
        context: The context in which the command is being invoked.
    """
    context.obj["group"] = "workspace"


def is_directory_traversal(directory: Union[str, Path]) -> bool:
    """Check for directory traversal.

    Args:
        directory (Union[str, Path]): The directory to check.

    Returns:
        bool: True if directory traversal is detected, False otherwise.
    """
    cwd = os.path.abspath(os.getcwd())
    requested_path = os.path.relpath(directory, start=cwd)
    requested_path = os.path.abspath(requested_path)
    common_prefix = os.path.commonprefix([requested_path, cwd])
    return common_prefix != cwd


def create_dirs(prefix):
    """Create workspace directories.

    Args:
        prefix: The prefix for the directories to be created.
    """

    echo("Creating Workspace Directories")

    (prefix / "cert").mkdir(parents=True, exist_ok=True)  # certifications
    (prefix / "data").mkdir(parents=True, exist_ok=True)  # training data
    (prefix / "logs").mkdir(parents=True, exist_ok=True)  # training logs
    (prefix / "save").mkdir(parents=True, exist_ok=True)  # model weight saves / initialization
    (prefix / "src").mkdir(parents=True, exist_ok=True)  # model code

    shutil.copyfile(WORKSPACE / "workspace" / ".workspace", prefix / ".workspace")


def create_temp(prefix, template):
    """Create workspace templates.

    Args:
        prefix: The prefix for the directories to be created.
        template: The template to use for creating the workspace.
    """

    src = template if os.path.isabs(template) else WORKSPACE / template
    echo(f"Creating Workspace Templates from {src} in {prefix}")
    shutil.copytree(
        src=src,
        dst=prefix,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )  # from template workspace


def get_templates():
    """Grab the default templates from the distribution.

    Returns:
        list: A list of default templates.
    """

    return [
        d.name
        for d in WORKSPACE.glob("*")
        if d.is_dir() and d.name not in ["__pycache__", "workspace", "experimental"]
    ]


@workspace.command(name="create")
@option("--prefix", required=True, help="Workspace name or path", type=ClickPath())
@option("--template", required=True, type=Choice(get_templates()))
def create_(prefix, template):
    """Create the workspace.

    Args:
        prefix: The prefix for the directories to be created.
        template: The template to use for creating the workspace.
    """
    if is_directory_traversal(prefix):
        echo("Workspace name or path is out of the openfl workspace scope.")
        sys.exit(1)
    create(prefix, template)


def create(prefix, template):
    """Create federated learning workspace.

    Args:
        prefix: The prefix for the directories to be created.
        template: The template to use for creating the workspace.
    """

    if not OPENFL_USERDIR.exists():
        OPENFL_USERDIR.mkdir()

    prefix = Path(prefix).absolute()

    create_dirs(prefix)
    create_temp(prefix, template)

    requirements_filename = "requirements.txt"

    if os.path.isfile(f"{str(prefix)}/{requirements_filename}"):
        check_call(
            [
                executable,
                "-m",
                "pip",
                "install",
                "-r",
                f"{prefix}/requirements.txt",
            ],
            shell=False,
        )
        echo(f"Successfully installed packages from {prefix}/requirements.txt.")
    else:
        echo("No additional requirements for workspace defined. Skipping...")
    prefix_hash = _get_dir_hash(str(prefix.absolute()))
    with open(
        OPENFL_USERDIR / f"requirements.{prefix_hash}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        check_call([executable, "-m", "pip", "freeze"], shell=False, stdout=f)

    apply_template_plan(prefix, template)

    print_tree(prefix, level=3)


@workspace.command(name="import")
@option(
    "--archive",
    required=True,
    help="Path to workspace archive.",
    type=ClickPath(exists=True),
)
def import_(archive):
    """Import a federated learning workspace generated by `fx workspace export`."""
    dir_path = os.path.basename(os.path.abspath(archive)).split(".")[0]
    shutil.unpack_archive(archive, extract_dir=dir_path)
    echo(f"Imported workspace `{archive}`.")
    echo("You may need to copy your PKI certificates to join the federation.")


@workspace.command(name="certify")
def certify_():
    """Create certificate authority for federation."""
    certify()


def certify():
    """Create certificate authority for federation."""

    echo("Setting Up Certificate Authority...\n")

    echo("1.  Create Root CA")
    echo("1.1 Create Directories")

    (CERT_DIR / "ca/root-ca/private").mkdir(parents=True, exist_ok=True, mode=0o700)
    (CERT_DIR / "ca/root-ca/db").mkdir(parents=True, exist_ok=True)

    echo("1.2 Create Database")

    with open(CERT_DIR / "ca/root-ca/db/root-ca.db", "w", encoding="utf-8") as f:
        pass  # write empty file
    with open(CERT_DIR / "ca/root-ca/db/root-ca.db.attr", "w", encoding="utf-8") as f:
        pass  # write empty file

    with open(CERT_DIR / "ca/root-ca/db/root-ca.crt.srl", "w", encoding="utf-8") as f:
        f.write("01")  # write file with '01'
    with open(CERT_DIR / "ca/root-ca/db/root-ca.crl.srl", "w", encoding="utf-8") as f:
        f.write("01")  # write file with '01'

    echo("1.3 Create CA Request and Certificate")

    root_crt_path = "ca/root-ca.crt"
    root_key_path = "ca/root-ca/private/root-ca.key"

    root_private_key, root_cert = generate_root_cert()

    # Write root CA certificate to disk
    with open(CERT_DIR / root_crt_path, "wb") as f:
        f.write(
            root_cert.public_bytes(
                encoding=serialization.Encoding.PEM,
            )
        )

    with open(CERT_DIR / root_key_path, "wb") as f:
        f.write(
            root_private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    echo("2.  Create Signing Certificate")
    echo("2.1 Create Directories")

    (CERT_DIR / "ca/signing-ca/private").mkdir(parents=True, exist_ok=True, mode=0o700)
    (CERT_DIR / "ca/signing-ca/db").mkdir(parents=True, exist_ok=True)

    echo("2.2 Create Database")

    with open(CERT_DIR / "ca/signing-ca/db/signing-ca.db", "w", encoding="utf-8") as f:
        pass  # write empty file
    with open(CERT_DIR / "ca/signing-ca/db/signing-ca.db.attr", "w", encoding="utf-8") as f:
        pass  # write empty file

    with open(CERT_DIR / "ca/signing-ca/db/signing-ca.crt.srl", "w", encoding="utf-8") as f:
        f.write("01")  # write file with '01'
    with open(CERT_DIR / "ca/signing-ca/db/signing-ca.crl.srl", "w", encoding="utf-8") as f:
        f.write("01")  # write file with '01'

    echo("2.3 Create Signing Certificate CSR")

    signing_csr_path = "ca/signing-ca.csr"
    signing_crt_path = "ca/signing-ca.crt"
    signing_key_path = "ca/signing-ca/private/signing-ca.key"

    signing_private_key, signing_csr = generate_signing_csr()

    # Write Signing CA CSR to disk
    with open(CERT_DIR / signing_csr_path, "wb") as f:
        f.write(
            signing_csr.public_bytes(
                encoding=serialization.Encoding.PEM,
            )
        )

    with open(CERT_DIR / signing_key_path, "wb") as f:
        f.write(
            signing_private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    echo("2.4 Sign Signing Certificate CSR")

    signing_cert = sign_certificate(signing_csr, root_private_key, root_cert.subject, ca=True)

    with open(CERT_DIR / signing_crt_path, "wb") as f:
        f.write(
            signing_cert.public_bytes(
                encoding=serialization.Encoding.PEM,
            )
        )

    echo("3   Create Certificate Chain")

    # create certificate chain file by combining root-ca and signing-ca
    with open(CERT_DIR / "cert_chain.crt", "w", encoding="utf-8") as d:
        with open(CERT_DIR / "ca/root-ca.crt", encoding="utf-8") as s:
            d.write(s.read())
        with open(CERT_DIR / "ca/signing-ca.crt") as s:
            d.write(s.read())

    echo("\nDone.")


def _get_dir_hash(path):
    """Get the hash of a directory.

    Args:
        path (str): The path of the directory.

    Returns:
        str: The hash of the directory.
    """
    hash_ = sha256()
    hash_.update(path.encode("utf-8"))
    hash_ = hash_.hexdigest()
    return hash_


# Commands for workspace packaging and distribution
# -------------------------------------------------


@workspace.command(name="export")
def export_() -> str:
    """
    Exports the OpenFL workspace (in current directory)
    to an archive.

    \b
    The archive contains the following files/dirs copied as-is:
        - `src`: All experiment source code.
        - `plan`: The FL plan directory.
        - `save`: Model initial weights.
        - `requirements.txt`: Package list required for the experiment.

    This archive does *not* copy `data`, `logs`, or secrets.

    This command takes no arguments.
    """
    plan_file = os.path.abspath(os.path.join("plan", "plan.yaml"))
    if not os.path.isfile(plan_file):
        raise FileNotFoundError(
            f"{plan_file} does not exist in the current directory.\n"
            "Please ensure this command is being run from a workspace."
        )
    plan.freeze_plan(plan_file)

    # Create a staging area.
    workspace_name = os.path.basename(os.getcwd())
    tmp_dir = os.path.join(tempfile.mkdtemp(), "openfl", workspace_name)
    ignore = shutil.ignore_patterns(
        *["__pycache__", "*.crt", "*.key", "*.csr", "*.srl", "*.pem", "*.pbuf"]
    )

    # Export the minimum required files to set up a collaborator
    # os.makedirs(os.path.join(tmp_dir, 'save'), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "data"), exist_ok=True)
    shutil.copytree("src", os.path.join(tmp_dir, "src"), ignore=ignore)
    shutil.copytree("plan", os.path.join(tmp_dir, "plan"), ignore=ignore)
    shutil.copytree("save", os.path.join(tmp_dir, "save"))
    shutil.copy2("requirements.txt", os.path.join(tmp_dir, "requirements.txt"))

    _ws_identifier_file = ".workspace"
    if not os.path.isfile(_ws_identifier_file):
        openfl_ws_identifier_file = os.path.join(WORKSPACE, "workspace", _ws_identifier_file)
        logging.warning(
            f"`{_ws_identifier_file}` is missing, " f"copying {openfl_ws_identifier_file} as-is."
        )
        shutil.copy2(openfl_ws_identifier_file, tmp_dir)
    shutil.copy2(_ws_identifier_file, tmp_dir)

    # Create Zip archive of directory
    _ARCHIVE_FORMAT = "zip"
    shutil.make_archive(workspace_name, _ARCHIVE_FORMAT, tmp_dir)
    archive = f"{workspace_name}.{_ARCHIVE_FORMAT}"
    logging.info(f"Export: {archive} created")
    return archive


@workspace.command(name="dockerize")
@option(
    "--save",
    is_flag=True,
    default=False,
    help="Export the docker image as <workspace_name>.tar file.",
)
@option(
    "--rebuild",
    is_flag=True,
    default=False,
    help="If set, rebuilds docker images with `--no-cache` option.",
)
@option(
    "--enclave-key",
    "enclave_key",
    type=str,
    required=False,
    help=(
        "Path to an enclave signing key. If not provided, a key will be auto-generated in the "
        "workspace. Note that this command builds a TEE-ready image, key is NOT packaged along "
        "with the image. You have the flexibility to not run inside a TEE later."
    ),
)
@option(
    "--revision",
    required=False,
    default=None,
    help=(
        "Optional, version of OpenFL source code to build base image from. "
        "If unspecified, default value in `Dockerfile.base` will be used, "
        "typically the latest stable release. "
        "Format: <OPENFL_GIT_URL>@<COMMIT_ID/BRANCH>"
    ),
)
@pass_context
def dockerize_(context, save: bool, rebuild: bool, enclave_key: str, revision: str):
    """Package current workspace as a TEE-ready Docker image."""

    # Docker build options
    options = []
    options.append("--no-cache" if rebuild else "")
    options.append(f"--build-arg OPENFL_REVISION={revision}" if revision else "")
    options = " ".join(options)

    # Export workspace
    archive = context.invoke(export_)
    workspace_name, _ = archive.split(".")

    # Build OpenFL base image.
    logging.info("Building OpenFL Base image")
    base_image_build_cmd = (
        "DOCKER_BUILDKIT=1 docker build {options} "
        "-t {image_name} "
        "-f {dockerfile} "
        "{build_context}"
    ).format(
        options=options,
        image_name="openfl",
        dockerfile=os.path.join(SITEPACKS, "openfl-docker", "Dockerfile.base"),
        build_context=".",
    )
    _execute(base_image_build_cmd)

    # Build workspace image.
    options = []
    options.append("--no-cache" if rebuild else "")
    options = " ".join(options)
    if enclave_key is None:
        _execute("openssl genrsa -out key.pem -3 3072")
        enclave_key = os.path.abspath("key.pem")
        logging.info(f"Generated new enclave key: {enclave_key}")
    else:
        enclave_key = os.path.abspath(enclave_key)
        if not os.path.exists(enclave_key):
            raise FileNotFoundError(f"Enclave key `{enclave_key}` does not exist")
        logging.info(f"Using enclave key: {enclave_key}")

    logging.info("Building workspace image")
    ws_image_build_cmd = (
        "DOCKER_BUILDKIT=1 docker build {options} "
        "--build-arg WORKSPACE_NAME={workspace_name} "
        "--secret id=signer-key,src={enclave_key} "
        "-t {image_name} "
        "-f {dockerfile} "
        "{build_context}"
    ).format(
        options=options,
        image_name=workspace_name,
        workspace_name=workspace_name,
        enclave_key=enclave_key,
        dockerfile=os.path.join(SITEPACKS, "openfl-docker", "Dockerfile.workspace"),
        build_context=".",
    )
    _execute(ws_image_build_cmd)

    # Export workspace as tarball (optional)
    if save:
        logging.info("Saving workspace docker image...")
        save_image_cmd = "docker save {image_name} -o {image_name}.tar"
        _execute(save_image_cmd.format(image_name=workspace_name))
        logging.info(f"Docker image saved to file: {workspace_name}.tar")


def apply_template_plan(prefix, template):
    """Copy plan file from template folder.

    This function unfolds default values from template plan configuration
    and writes the configuration to the current workspace.

    Args:
        prefix: The prefix for the directories to be created.
        template: The template to use for creating the workspace.
    """

    template_plan = Plan.parse(WORKSPACE / template / "plan" / "plan.yaml")

    Plan.dump(prefix / "plan" / "plan.yaml", template_plan.config)


def _execute(cmd: str, verbose=True) -> None:
    """Executes `cmd` as a subprocess

    Args:
        cmd (str): Command to be executed.

    Raises:
        Exception: If return code is nonzero

    Returns:
        `stdout` of the command as list of messages
    """
    logging.info(f"Executing: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    stdout_log = []
    for line in process.stdout:
        msg = line.rstrip().decode("utf-8")
        stdout_log.append(msg)
        if verbose:
            logging.info(msg)

    process.communicate()
    if process.returncode != 0:
        raise Exception(f"`{cmd}` failed with return code {process.returncode}")

    return stdout_log
