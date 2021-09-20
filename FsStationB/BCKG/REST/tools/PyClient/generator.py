import click

from openapi_parser.exporter import PackageWriter
from openapi_parser.parser.loader import OpenApiParser

@click.command()
@click.option(
    "--source",
    help="REST API .yml definition file",
    default='../../../bckg_api.yml',
    required=False,
)
@click.option(
    "--pyclient",
    help="Target directory for Python client library",
    default='../../src/PyClient',
    required=False,
)
def main(source, pyclient):
    parser = OpenApiParser.open(source)
    parser.load_all()

    package_writer = PackageWriter(parser, destination_dir=pyclient)
    package_writer.write_package(clean=True)

    return 0

if (__name__ == '__main__'):
    exit_code = main()
    exit(exit_code)
