import hashlib
import click


@click.command()
@click.option("-m", "--model-path", type=str, required=True)
def main(model_path):
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for buffer in iter(lambda: f.read(8192), b""):
            sha256.update(buffer)
        print(sha256.hexdigest()[:8])


if __name__ == "__main__":
    main()
