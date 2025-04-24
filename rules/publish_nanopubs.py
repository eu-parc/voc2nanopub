#!/usr/bin/env python
"""
Nanopub Batch Upload GitHub Action Script

TODO:
- add ability to restrict model to NamedThing if graph is too large
"""

import sys
import click
import nanopub
import nanopub.definitions
import rdflib
import logging

from collections import defaultdict, deque
from pathlib import Path
from typing import List, Optional, Mapping

from linkml_runtime.utils.schemaview import SchemaView
from linkml_runtime.loaders import YAMLLoader
from linkml_runtime.dumpers import YAMLDumper
from linkml_runtime.dumpers import RDFLibDumper
from linkml.generators.pythongen import PythonGenerator
from linkml_runtime.utils.yamlutils import YAMLRoot

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


BASE_NAMESPACE = rdflib.Namespace("https://w3id.org/peh/peh-model")


def topological_sort(
    objects: List[YAMLRoot], id_key: str, parent_key: str
) -> List[YAMLRoot]:
    if parent_key is None:
        return objects
    # Build adjacency list and in-degree map
    adj_list = defaultdict(list)
    in_degree = defaultdict(int)
    obj_map = {getattr(obj, id_key): obj for obj in objects}

    for obj in objects:
        parent = getattr(obj, parent_key, None)
        if parent is not None:
            adj_list[parent].append(getattr(obj, id_key))
            in_degree[getattr(obj, id_key)] += 1

    # Collect nodes with no incoming edges (in-degree 0)
    queue = deque()
    for obj_id in obj_map:
        if in_degree[obj_id] == 0:
            queue.append(obj_id)

    # Perform topological sort
    sorted_ids = []
    while queue:
        current = queue.popleft()
        sorted_ids.append(current)

        for child in adj_list[current]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # Return sorted objects
    return [obj_map[obj_id] for obj_id in sorted_ids]


def load_yaml(
    schema_path: str,
    data_path: str,
) -> YAMLRoot:
    try:
        # Load schema
        schema_path = Path(schema_path)
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        schema_view = SchemaView(str(schema_path))

        # Generate Python classes from schema
        python_module = PythonGenerator(str(schema_path)).compile_module()
        target_class = "EntityList"  # This could be made configurable

        if target_class not in python_module.__dict__:
            raise ValueError(f"Target class '{target_class}' not found in schema")

        py_target_class = python_module.__dict__[target_class]

        # Load data
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # data is instance of EntityList dataclass
        data = YAMLLoader().load(str(data_path), py_target_class)

        return data, schema_view

    except Exception as e:
        logger.error(f"Error in load_yaml: {e}")
        raise


def process_yaml_root(
    root: YAMLRoot,
    target_name: str,
    id_key: str = "id",
    parent_key: Optional[str] = None,
) -> List:
    try:
        # Process entities
        target_data_list = getattr(root, target_name, None)
        if target_data_list is None:
            raise ValueError(f"Target list '{target_name}' not found in data dict")

        # topological sort of data
        if parent_key is not None:
            target_data_list = topological_sort(
                target_data_list, id_key=id_key, parent_key=parent_key
            )

        return target_data_list

    except Exception as e:
        logger.error(f"Error in process_yaml_root: {e}")
        raise


def get_property_mapping(
    data: List, schema_view: SchemaView, base: rdflib.Namespace
) -> Mapping:
    """
    Mapping of the kind: {property_name: slot_uri}
    example: {'name': rdflib.term.URIRef('http://www.w3.org/2004/02/skos/core#altLabel')}
    """
    namespace_mapping = {}
    for entity in data:
        if getattr(entity, "translations") is not None:
            for translation in entity.translations:
                if translation.property_name not in namespace_mapping:
                    property_name = translation.property_name
                    slot_def = schema_view.all_slots().get(property_name)
                    curie_str = getattr(slot_def, "slot_uri")
                    if curie_str is None:
                        curie_str = base[property_name]
                    uri_str = schema_view.expand_curie(curie_str)
                    namespace_mapping[property_name] = rdflib.term.URIRef(uri_str)

    return namespace_mapping


def add_translation_to_graph(
    g: rdflib.Graph, property_mapping: Mapping
) -> rdflib.Graph:
    if len(property_mapping) == 0:
        logger.info("LinkML schema does not contain translations.")
        return g

    #  Iterate over the triples and perform the transformation and removal
    for s, _, o in g.triples((None, BASE_NAMESPACE.translations, None)):
        language = g.value(o, BASE_NAMESPACE.language)
        property_name = str(g.value(o, BASE_NAMESPACE.property_name))
        translated_value = g.value(o, BASE_NAMESPACE.translated_value)
        # Apply the mapping
        if property_name in property_mapping:
            mapped_property = property_mapping[property_name]
            g.add((s, mapped_property, rdflib.Literal(translated_value, lang=language)))

        # Remove the unnecessary blank node triples
        g.remove((o, None, None))
        g.remove((None, None, o))

    return g


def is_nanopub_id(key: str):
    allowed_prefixes = [
        "http://purl.org",
        "https://purl.org",
        "http://w3id.org",
        "https://w3id.org",
    ]
    for prefix in allowed_prefixes:
        if key.startswith(prefix):
            return True
    return False


def check_nanopub_existence(np_conf: nanopub.NanopubConf, entity: YAMLRoot) -> bool:
    try:
        url = getattr(entity, "id", None)
        if url is not None:
            return is_nanopub_id(url)
        else:
            raise ValueError("Entity id is None.")

    except Exception as e:
        logger.error(f"Error in check_nanopub_existence: {e}")


def yaml_dump(root: YAMLRoot, target_name: str, entities: List, file_name: str):
    # Use setattr to update the target field
    setattr(root, target_name, entities)
    return YAMLDumper().dump(root, to_file=file_name)


def is_valid_assertion_graph(g: rdflib.Graph) -> bool:
    # TODO: add more checks
    return len(g) > 0


def build_rdf_graph(
    entity: "YAMLRoot",
    schema_view: SchemaView,
    translation_namespace_mapping: Optional[Mapping] = None,
) -> rdflib.Graph:
    """
    Convert a LinkML entity to an RDF graph.

    Args:
        entity: The LinkML entity to convert
        schema_view: The schema view defining the entity structure

    Returns:
        An RDF graph representing the entity
    """
    try:
        rdf_string = RDFLibDumper().dumps(entity, schema_view)
        g = rdflib.Graph()
        g.parse(data=rdf_string)
        assert len(g) < nanopub.definitions.MAX_TRIPLES_PER_NANOPUB
        # ADD TRANSLATION
        if translation_namespace_mapping is not None:
            g = add_translation_to_graph(g, translation_namespace_mapping)

        if is_valid_assertion_graph(g):
            return g
        else:
            raise AssertionError("Assertion Graph is invalid.")
    except Exception as e:
        logger.error(f"Error converting entity to RDF: {e}")
        raise


@click.command()
@click.option(
    "--schema",
    "-s",
    "schema_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the LinkML schema file",
)
@click.option(
    "--data",
    "-d",
    "data_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the YAML data file",
)
@click.option(
    "--target",
    "-t",
    "target_name",
    required=True,
    help="Name of the target entity list in the data file",
)
@click.option(
    "--orcid-id",
    required=True,
    envvar="NANOPUB_ORCID_ID",
    help="ORCID ID for nanopub profile",
)
@click.option(
    "--name", required=True, envvar="NANOPUB_NAME", help="Name for nanopub profile"
)
@click.option(
    "--private-key",
    required=True,
    envvar="NANOPUB_PRIVATE_KEY",
    help="Private key for nanopub profile",
)
@click.option(
    "--public-key",
    required=True,
    envvar="NANOPUB_PUBLIC_KEY",
    help="Public key for nanopub profile",
)
@click.option(
    "--intro-nanopub-uri",
    required=True,
    envvar="NANOPUB_INTRO_URI",
    help="Introduction nanopub URI",
)
@click.option(
    "--test-server/--production-server",
    default=True,
    help="Use test server (default) or production server",
)
@click.option("--dry-run", is_flag=True, help="Prepare nanopubs but do not publish")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--parent-key",
    required=False,
    help="Name of the field that references parent entities (for topological sorting)",
    default=None,
)
@click.option(
    "--output",
    "output_path",
    required=False,
    type=click.Path(),
    help="Path to output YAML data file",
    default=None,
)
def main(
    schema_path: str,
    data_path: str,
    target_name: str,
    parent_key: str,
    orcid_id: str,
    name: str,
    private_key: str,
    public_key: str,
    intro_nanopub_uri: str,
    test_server: bool = True,
    dry_run: bool = False,
    verbose: bool = False,
    output_path: str = None,
):
    """
    Create and publish nanopublications from structured data.

    This tool takes data structured according to a LinkML schema and publishes
    it as nanopublications. It's designed to be run as part of a GitHub Actions
    workflow, with authentication details provided as GitHub secrets.
    """
    # Set logging level based on verbose flag
    if verbose:
        logger.setLevel(logging.DEBUG)

    try:
        # Create nanopub profile
        profile = nanopub.Profile(
            orcid_id=orcid_id,
            name=name,
            private_key=private_key,
            public_key=public_key,
            introduction_nanopub_uri=intro_nanopub_uri,
        )

        # Create nanopub configuration
        np_conf = nanopub.NanopubConf(
            profile=profile,
            use_test_server=test_server,
            add_prov_generated_time=True,
            attribute_publication_to_profile=True,
        )

        logger.info(f"Processing data from {data_path} using schema {schema_path}")

        # Count for reporting
        processed = 0
        published = 0
        skipped = 0

        # load data
        yaml_root, schema_view = load_yaml(schema_path, data_path)
        entities = process_yaml_root(
            yaml_root, target_name, id_key="id", parent_key=parent_key
        )
        id_map = {}

        namespace_mapping = get_property_mapping(entities, schema_view, BASE_NAMESPACE)
        if len(namespace_mapping) == 0:
            namespace_mapping = None
        # Process each entity and publish as nanopub
        for entity in entities:
            np_uri = None
            graph = build_rdf_graph(entity, schema_view, namespace_mapping)
            processed += 1

            # Check if this nanopub already exists
            if check_nanopub_existence(np_conf, entity):
                logger.info(f"Nanopub {processed} already exists, skipping")
                skipped += 1
                continue

            # Create nanopub
            np = nanopub.Nanopub(conf=np_conf, assertion=graph)
            np.sign()
            logger.info(f"Nanopub {processed} signed")
            np_uri = np.metadata.np_uri
            if np_uri is None:
                raise ValueError("no URI returned by nanpub server.")

            ## modify parent key
            # check if parent_key has been set
            if parent_key is not None:
                old_parent_key = getattr(entity, parent_key)
                # check if a particular entity has a parent_key field
                if old_parent_key is not None:
                    if not is_nanopub_id(old_parent_key):
                        new_parent_key = id_map.get(old_parent_key, None)
                        if new_parent_key is None:
                            raise AssertionError("new_parent_key is None.")
                        setattr(entity, parent_key, new_parent_key)

                old_id = getattr(entity, "id")
                id_map[old_id] = np_uri

            # modify entity uri
            setattr(entity, "id", np_uri)
            logger.info(f"URI generated: {np_uri}")

            if not dry_run:
                publication_info = np.publish()
                published += 1
                logger.info(f"Nanopub {processed} published: {publication_info}")

        # Report summary
        logger.info(
            f"Nanopub processing complete. Processed: {processed}, "
            f"Published: {published}, Skipped: {skipped}"
        )

        if output_path is None:
            output_path = data_path
        return yaml_dump(yaml_root, target_name, entities, output_path)

    except Exception as e:
        logger.error(f"Error in nanopub processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
