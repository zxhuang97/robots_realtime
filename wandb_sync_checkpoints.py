import argparse
import os
import sys
import wandb
from pathlib import Path

def upload_checkpoint(
    path: str,
    project: str,
    entity: str,
    artifact_name: str,
    artifact_type: str = "model",
    description: str = None,
    aliases: list[str] = None
):
    """
    Uploads a local directory as a W&B artifact.
    """
    path = Path(path).resolve()
    if not path.exists():
        print(f"Error: Path '{path}' does not exist.")
        sys.exit(1)

    # Use the artifact name as the run name for easier identification, 
    # but append a timestamp or similar if needed to avoid collisions 
    # if we were tracking metrics. For uploads, collisions are fine/handled.
    # actually, let's just let W&B pick a fun name, or use "upload-{artifact_name}"
    run_name = f"upload-{artifact_name}"

    print(f"Initializing W&B run for upload...")
    run = wandb.init(
        project=project,
        entity=entity,
        job_type="upload-checkpoint",
        name=run_name,
        # id=run_id, # Removed complexity
        resume="allow"
    )

    print(f"Creating artifact '{artifact_name}' (type: {artifact_type})...")
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=description
    )

    if path.is_dir():
        print(f"Adding directory {path} to artifact...")
        artifact.add_dir(str(path))
    else:
        print(f"Adding file {path} to artifact...")
        artifact.add_file(str(path))

    print("Logging artifact (this starts the upload)...")
    run.log_artifact(artifact, aliases=aliases or [])
    
    # Wait for the artifact to finish uploading
    print("Waiting for upload to complete...")
    artifact.wait()
    
    run.finish()
    print(f"Successfully uploaded to {entity}/{project}/{artifact_name}")


def download_checkpoint(
    path: str,
    project: str,
    entity: str,
    artifact_name: str,
    version: str = "latest"
):
    """
    Downloads a W&B artifact to a local directory.
    """
    print(f"Initializing W&B run for download...")
    run = wandb.init(
        project=project,
        entity=entity,
        job_type="download-checkpoint",
        name=f"download-{artifact_name}"
    )

    artifact_path = f"{entity}/{project}/{artifact_name}:{version}"
    print(f"Downloading artifact '{artifact_path}'...")
    
    try:
        artifact = run.use_artifact(artifact_path)
        download_dir = artifact.download(root=path)
        print(f"Successfully downloaded artifact to '{download_dir}'")
    except Exception as e:
        print(f"Error downloading artifact: {e}")
        sys.exit(1)
    finally:
        run.finish()

def main():
    parser = argparse.ArgumentParser(
        description="Sync checkpoints (upload/download) using Weights & Biases Artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a checkpoint folder
  python wandb_sync_checkpoints.py upload ./checkpoints/my_model --project my-project --entity my-team --name my-model-checkpoint

  # Download the latest version of a checkpoint
  python wandb_sync_checkpoints.py download ./downloaded_checkpoints --project my-project --entity my-team --name my-model-checkpoint

  # Download a specific version
  python wandb_sync_checkpoints.py download ./downloaded_checkpoints --project my-project --entity my-team --name my-model-checkpoint --version v3
"""
    )

    parser.add_argument(
        "action",
        choices=["upload", "download"],
        help="Action to perform: upload or download."
    )
    parser.add_argument(
        "path",
        help="Local path. For upload: source directory/file. For download: destination directory."
    )
    parser.add_argument(
        "--project",
        default="flow_checkpoints",
        help="W&B project name. Defaults to WANDB_PROJECT env var."
    )
    parser.add_argument(
        "--entity",
        default="far-wandb",
        help="W&B entity (username or team name). Defaults to WANDB_ENTITY env var."
    )
    parser.add_argument(
        "--name",
        dest="artifact_name",
        help="Name of the artifact. Defaults to the directory name if not provided."
    )
    parser.add_argument(
        "--type",
        dest="artifact_type",
        default="model",
        help="Type of the artifact (default: model). Used only for upload."
    )
    parser.add_argument(
        "--version",
        default="latest",
        help="Artifact version or alias to download (default: latest). Used only for download."
    )
    parser.add_argument(
        "--description",
        help="Description for the artifact. Used only for upload."
    )
    # Removed --run-name and --run-id to simplify
    parser.add_argument(
        "--alias",
        action="append",
        dest="aliases",
        help="Alias for the artifact (can be used multiple times). Used only for upload."
    )

    args = parser.parse_args()

    if not args.project:
        parser.error("The --project argument or WANDB_PROJECT environment variable is required.")
    if not args.entity:
        parser.error("The --entity argument or WANDB_ENTITY environment variable is required.")

    # Determine artifact name from path if not provided
    if not args.artifact_name:
        args.artifact_name = os.path.basename(os.path.normpath(args.path))

    # Check for WANDB_API_KEY or login
    if "WANDB_API_KEY" not in os.environ:
        # We can't easily check if they are logged in via 'wandb login' programmatically without side effects
        # but wandb.init will throw/prompt if not authenticated.
        pass

    if args.action == "upload":
        upload_checkpoint(
            path=args.path,
            project=args.project,
            entity=args.entity,
            artifact_name=args.artifact_name,
            artifact_type=args.artifact_type,
            description=args.description,
            aliases=args.aliases
        )
    elif args.action == "download":
        download_checkpoint(
            path=args.path,
            project=args.project,
            entity=args.entity,
            artifact_name=args.artifact_name,
            version=args.version,
            # run_name=args.run_name # Removed
        )

if __name__ == "__main__":
    main()

