"""Petit script pour voir ce que renvoie l'API BioPortal quand on cherche un terme dans SNOMED CT.

Lancement, depuis la racine du projet :
    py -3.11 src/verification/voir_api.py "white blood cell count"
"""

import json
import os
import sys
import urllib.parse
import urllib.request
from dotenv import load_dotenv

# 1. Le terme à chercher : celui écrit après le nom du script, sinon un exemple
term = "white blood cell count"
if len(sys.argv) > 1:
    term = sys.argv[1]

# 2. La clé API, lue dans le fichier .env (elle n'est jamais affichée)
load_dotenv()
key = os.environ["BIOPORTAL_API_KEY"]

# 3. L'adresse de la recherche : le terme, l'ontologie SNOMED CT, 5 résultats au maximum
params = urllib.parse.urlencode({
    "q": term,
    "ontologies": "SNOMEDCT",
    "pagesize": 5,
    # "require_exact_match": "true",  # enlever le # pour une recherche exacte
})
url = "https://data.bioontology.org/search?" + params
print("Adresse appelée :", url)

# 4. Envoi de la requête (la clé passe dans l'en-tête) et lecture de la réponse JSON
request = urllib.request.Request(url, headers={"Authorization": "apikey token=" + key})
response = urllib.request.urlopen(request)
data = json.load(response)

# 5. Ce que l'API renvoie : le nombre de résultats, puis les 5 premiers concepts
print("Nombre total de résultats :", data["totalCount"])
print()
for concept in data["collection"]:
    print("Adresse du concept :", concept["@id"])
    print("Nom préféré        :", concept.get("prefLabel"))
    print("Synonymes          :", concept.get("synonym"))
    print("Définition         :", concept.get("definition"))
    print()

# 6. Le premier concept en entier, exactement comme l'API l'envoie
if data["collection"]:
    print("=== Réponse brute du premier concept ===")
    print(json.dumps(data["collection"][0], indent=2, ensure_ascii=False))
