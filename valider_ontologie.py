from rdflib import Graph

g = Graph()
g.parse("questionnaire_ontology.ttl", format="turtle")

print("RDF/Turtle parsed successfully.")
print("Number of triples:", len(g))