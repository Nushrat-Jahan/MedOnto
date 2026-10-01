from rdflib import Graph

g = Graph()
g.parse("fragment_ontologies_version2.ttl", format="turtle")

print("RDF/Turtle parsed successfully.")
print("Number of triples:", len(g))