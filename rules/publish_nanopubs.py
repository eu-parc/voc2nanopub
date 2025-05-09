#!/usr/bin/env python
"""
Nanopub Batch Upload GitHub Action Script

TODO:
- add ability to restrict model to NamedThing if graph is too large
- figure out when default namespace is used and when model id
"""

import sys
import click
import hashlib
import logging
import nanopub
import nanopub.definitions
import pathlib
import rdflib
import re
import requests

from collections import defaultdict, deque
from pathlib import Path
from rdflib.namespace import SKOS, RDF
from uuid import uuid4
from typing import List, Optional, Mapping, Set

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


BASE_NAMESPACE = rdflib.Namespace("https://w3id.org/peh/terms/")
PEH_NAMESPACE = "https://w3id.org/peh/"


class IdentifierGenerator:
    def __init__(
        self, type_prefix: Optional[str] = None, namespace: str = PEH_NAMESPACE
    ):
        self.namespace = namespace
        self.type_prefix = type_prefix
        self.registered_ids: Set[str] = set()

    def is_id_available(self, identifier: str) -> bool:
        if identifier in self.registered_ids:
            return False

        return True

    def is_namespace_id(self, key: str):
        if key.startswith(self.namespace):
            return True
        return False

    def register_id(self, identifier: str) -> None:
        self.registered_ids.add(identifier)

    def generate_id(
        self,
        term_name: str,
        method: str = "hash",
        check_collision: bool = True,
        max_attempts: int = 10,
    ) -> str:
        """
        Generate a structured identifier for a vocabulary term with collision detection.

        Args:
            term_name: The name of the term
            method: ID generation method ('uuid', 'hash', 'sequential', or 'slug')
            check_collision: Whether to check for collisions
            max_attempts: Maximum number of attempts to generate a unique ID

        Returns:
            A structured identifier that's unique and available
        """
        attempts = 0

        while attempts < max_attempts:
            # Generate unique part based on method
            if method == "uuid":
                unique_part = str(uuid4())[:8]  # First 8 chars of UUID

            elif method == "hash":
                # Create a hash of the term name, possibly with a salt for retry attempts
                if attempts > 0:
                    salted_name = f"{term_name}-attempt-{attempts}"
                else:
                    salted_name = term_name

                hash_obj = hashlib.md5(salted_name.encode())
                unique_part = hash_obj.hexdigest()[:10]

            else:
                raise ValueError(
                    f"Unknown method: {method}. Available methods: uuid, hash, sequential, slug"
                )

            # Construct the full identifier
            if self.type_prefix is None:
                identifier = f"{self.namespace}{unique_part}"
            else:
                identifier = f"{self.namespace}{self.type_prefix}-{unique_part}"

            # Check if this identifier is available
            if check_collision:
                if self.is_id_available(identifier):
                    # Register the ID as used
                    self.register_id(identifier)
                    return identifier
            else:
                return identifier

            # If we got here, there was a collision - try again
            attempts += 1

        # If we exhausted all attempts, raise an error
        raise RuntimeError(
            f"Could not generate a unique identifier for '{term_name}' after {max_attempts} attempts"
        )


class NanopubGenerator:
    def __init__(
        self,
        orcid_id: str,
        name: str,
        private_key: str,
        public_key: str,
        intro_nanopub_uri: str,
        test_server: bool,
    ):
        self.profile = nanopub.Profile(
            orcid_id=orcid_id,
            name=name,
            private_key=private_key,
            public_key=public_key,
            introduction_nanopub_uri=intro_nanopub_uri,
        )

        self.np_conf = nanopub.NanopubConf(
            profile=self.profile,
            use_test_server=test_server,
            add_prov_generated_time=True,
            attribute_publication_to_profile=True,
        )

    def create_nanopub(self, assertion: rdflib.Graph) -> nanopub.Nanopub:
        return nanopub.Nanopub(conf=self.np_conf, assertion=assertion)

    def update_nanopub(self, np_uri: str, assertion: rdflib.Graph) -> nanopub.Nanopub:
        new_np = nanopub.NanopubUpdate(
            uri=np_uri,
            conf=self.np_conf,
            assertion=assertion,
        )
        new_np.sign()
        return new_np

    @classmethod
    def is_nanopub_id(cls, key: str):
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

    def check_nanopub_existence(self, entity: YAMLRoot) -> bool:
        try:
            # np_conf = self.np_conf
            url = getattr(entity, "id", None)
            if url is not None:
                return self.is_nanopub_id(url)
            else:
                raise ValueError("Entity id is None.")

        except Exception as e:
            logger.error(f"Error in check_nanopub_existence: {e}")


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
    try:
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
                g.add(
                    (
                        s,
                        mapped_property,
                        rdflib.Literal(translated_value, lang=language),
                    )
                )

            # Remove the unnecessary blank node triples
            g.remove((o, None, None))
            g.remove((None, None, o))

        return g

    except Exception as e:
        logging.error(f"Error in add_translation_to_graph: {e}")
        raise


def add_vocabulary_membership(
    g: rdflib.Graph, vocab_uri: str, subject_type: rdflib.URIRef
) -> rdflib.Graph:
    """
    Adds vocabulary membership information to each concept in the graph.

    Args:
        g: An rdflib Graph instance containing vocabulary terms
        vocab_uri: URI string of the vocabulary collection

    Returns:
        The modified graph with vocabulary membership added
    """
    try:
        # Create a URI reference for the vocabulary
        vocabulary = rdflib.URIRef(vocab_uri)
        concepts = list(g.subjects(RDF.type, subject_type))
        SKOS_COLLECTION = SKOS.inScheme
        # Add the membership triple to each concept
        for concept in concepts:
            g.add((concept, SKOS_COLLECTION, vocabulary))

        return g
    except Exception as e:
        logging.error(f"Error in add_vocabulary_membership: {e}")
        raise


def yaml_dump(root: YAMLRoot, target_name: str, entities: List, file_name: str):
    # Use setattr to update the target field
    setattr(root, target_name, entities)
    return YAMLDumper().dump(root, to_file=file_name)


def extract_id(url: str, type_prefix: Optional[str] = None):
    """Extract the type prefix (MA, UN, etc.) and the ID from a w3id.org URL."""
    match = re.search(rf"w3id\.org/peh/{type_prefix}-([a-f0-9]+)", url)
    if match:
        return match.group(1)
    return None


def generate_htaccess(redirects: List, type_prefix: str):
    """Generate .htaccess content."""

    rules = []

    for source, target in redirects:
        local_path = extract_id(source, type_prefix)
        if local_path:
            rules.append(f"RewriteRule ^{local_path}$ {target} [R=302,L]")

    return "\n".join(rules)


def update_htaccess(
    redirects: List, output_file: str, type_prefix: Optional[str] = None
):
    # example header
    # """Generate or update an .htaccess file."""
    # header = """RewriteEngine On
    #
    ## PEH redirections
    ## Format: Local ID to nanopub
    # """

    if not redirects:
        print("No valid redirects found in input file.", file=sys.stderr)
        sys.exit(1)

    new_content = generate_htaccess(redirects, type_prefix=type_prefix)

    with open(output_file, "w") as f:
        f.write(new_content)

    print(f"Successfully wrote .htaccess to {output_file}")
    print(f"Added {len(redirects)} redirect rules")


def dump_identifier_pairs(pairs: List[tuple], file_name: str):
    try:
        with open(file_name, "w") as outfile:
            for pair in pairs:
                w3id_uri, nanopub_uri = pair
                print(f"{w3id_uri}, {nanopub_uri}", file=outfile)
    except Exception as e:
        logging.error(f"Error in dump_identifier_pairs: {e}")
        raise


def is_valid_assertion_graph(g: rdflib.Graph) -> bool:
    # TODO: add more checks
    return len(g) > 0


def build_rdf_graph(
    entity: "YAMLRoot",
    schema_view: SchemaView,
    translation_namespace_mapping: Optional[Mapping] = None,
    vocab_uri: Optional[str] = None,
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
        # ADD vocabulary membership
        if vocab_uri is not None:
            entity_class_name = entity.__class__.__name__
            # example: subject_type = SKOS.Concept
            class_curie = schema_view.get_uri(entity_class_name)
            class_uri = schema_view.expand_curie(class_curie)
            subject_type = rdflib.term.URIRef(class_uri)
            g = add_vocabulary_membership(
                g, vocab_uri=vocab_uri, subject_type=subject_type
            )
        # ADD TRANSLATION
        if translation_namespace_mapping is not None:
            g = add_translation_to_graph(g, translation_namespace_mapping)
        if is_valid_assertion_graph(g):
            return g
        else:
            raise AssertionError("Assertion Graph is invalid.")
    except Exception as _:
        logger.error("Error converting entity to RDF:", exc_info=True)
        logger.debug("Entity details: %s", entity)
        logger.debug(
            "Additional context: vocab_uri=%s, translation_namespace_mapping=%s",
            vocab_uri,
            translation_namespace_mapping,
        )
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
@click.option(
    "--output-pairs",
    "output_path_pairs",
    required=False,
    type=click.Path(),
    help="Path to output identifier nanopub pairs",
    default=None,
)
@click.option(
    "--vocab",
    "vocab_uri",
    required=False,
    type=str,
    help="URI for the larger vocabulary this term is part of.",
    default=None,
)
@click.option(
    "--type-prefix",
    "type_prefix",
    required=False,
    type=str,
    default=None,
    help="Vocabulary-specific prefix for uri to be generated.",
)
@click.option(
    "--preflabel",
    "preflabel",
    required=True,
    type=str,
    help="Key to human readable identifier field for resource.",
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
    preflabel: str,
    test_server: bool = True,
    dry_run: bool = False,
    verbose: bool = False,
    output_path: str = None,
    output_path_pairs: str = None,
    vocab_uri: str = None,
    type_prefix: str = None,
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
        id_map = {}
        identifier_pairs = []
        # Count for reporting
        processed = 0
        published = 0
        updated = 0

        nanopub_generator = NanopubGenerator(
            orcid_id=orcid_id,
            name=name,
            private_key=private_key,
            public_key=public_key,
            intro_nanopub_uri=intro_nanopub_uri,
            test_server=test_server,
        )
        id_generator = IdentifierGenerator(type_prefix=type_prefix)

        logger.info(f"Processing data from {data_path} using schema {schema_path}")

        # load data
        yaml_root, schema_view = load_yaml(schema_path, data_path)
        entities = process_yaml_root(
            yaml_root, target_name, id_key="id", parent_key=parent_key
        )
        for entity in entities:
            # registring ids to prevent collisions
            current_id = getattr(entity, "id")
            if id_generator.is_namespace_id(current_id):
                id_generator.register_id(current_id)

        # make namespace mapping for language annotation purposes
        namespace_mapping = get_property_mapping(entities, schema_view, BASE_NAMESPACE)
        if len(namespace_mapping) == 0:
            namespace_mapping = None
        # Process each entity, generate identifier and publish nanopub
        for entity in entities:
            current_id = getattr(entity, "id")
            if id_generator.is_namespace_id(current_id):
                ## Entity is already published as nanopub
                logger.info(f"Entity {current_id} already exists, skipping")
                graph = build_rdf_graph(
                    entity, schema_view, namespace_mapping, vocab_uri=vocab_uri
                )
                # Fetch existing nanopub
                existing_graph = None
                # Compare new and existing graph
                if existing_graph != graph:
                    # use redirect to get nanopub uri
                    response = requests.get(
                        current_id, allow_redirects=True, timeout=10
                    )
                    if not response.status_code == 200:
                        logger.error(
                            f"Voc entry {current_id} could not be redirected for update."
                        )
                        continue
                    current_np_uri = response.url
                    new_np = nanopub_generator.update_nanopub(current_np_uri, graph)
                    new_np_uri = new_np.metadata.np_uri
                    logger.info(
                        f"Voc entry {current_id} has been updated. Updating Nanopub: {new_np_uri}"
                    )

                    if not dry_run:
                        publication_info = new_np.publish()
                        published += 1
                        logger.info(f"Nanopub update info: {publication_info}")

                    # create w3id - nanopub pairs
                    identifier_pairs.append((current_id, new_np_uri))
                    updated += 1

            else:
                ## Entity has no nanopub yet
                # create identifier
                peh_uri = id_generator.generate_id(getattr(entity, preflabel))

                ## modify parent key
                # check if parent_key has been set
                old_id = getattr(entity, "id")
                if parent_key is not None:
                    old_parent_key_value = getattr(entity, parent_key)
                    # check if a particular entity has a parent_key field
                    if old_parent_key_value is not None:
                        if not id_generator.is_namespace_id(old_parent_key_value):
                            new_parent_key_value = id_map.get(
                                old_parent_key_value, None
                            )
                            if new_parent_key_value is None:
                                raise AssertionError(
                                    "Parent key was not found in id_map."
                                )
                            new_parent_key_value_entity = getattr(
                                entity, parent_key
                            ).__class__(new_parent_key_value)
                            setattr(entity, parent_key, new_parent_key_value_entity)

                    id_map[old_id] = peh_uri

                # modify entity uri
                setattr(entity, "id", peh_uri)
                logger.info(f"URI generated: {peh_uri} for entity: {old_id}")

                graph = build_rdf_graph(
                    entity, schema_view, namespace_mapping, vocab_uri=vocab_uri
                )
                serialized_graph = graph.serialize()
                logger.info(f"nanopub statement: {serialized_graph}")
                processed += 1

                np = nanopub_generator.create_nanopub(assertion=graph)
                np.sign()
                logger.info(f"Nanopub {processed} signed")
                np_uri = np.metadata.np_uri
                if np_uri is None:
                    raise ValueError("no URI returned by nanpub server.")

                logger.info(f"Nanopub signed: {np_uri} for entity: {peh_uri}")

                if not dry_run:
                    publication_info = np.publish()
                    published += 1
                    logger.info(f"Nanopub {processed} published: {publication_info}")

                # create w3id - nanopub pairs
                identifier_pairs.append((peh_uri, np_uri))

        # Report summary
        logger.info(
            f"Processing complete. Processed: {processed}, "
            f"Published: {published}, Updated: {updated}"
        )

        if output_path is None:
            output_path = data_path
        _ = yaml_dump(yaml_root, target_name, entities, output_path)

        # dump identifier_pairs
        if output_path_pairs is None:
            output_path_pairs = "./pairs.txt"
        output_path_pairs = pathlib.Path(output_path_pairs).resolve()
        _ = update_htaccess(identifier_pairs, output_path_pairs, type_prefix)

    except Exception as e:
        logger.error(f"Error in processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
