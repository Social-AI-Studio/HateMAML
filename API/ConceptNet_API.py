""" ***** Warning *****

    The Web API has rate restriction.
    On an average we can ask for one query a second

    """


import requests


class ConceptNet:
    def __init__(self, lan):
        self.lan = lan
        self.web_api = "http://api.conceptnet.io/query"

    def lookup(self, term, verbose, limit=100):

        term = term.lower()

        if " " in term:
            term = term.replace(" ", "_")
        url_to_search = (
            self.web_api
            + "?start=/c/"
            + self.lan
            + "/"
            + term
            + "&?offset=0&limit=="
            + str(limit)
        )
        data = requests.get(url_to_search).json()
        if verbose:
            print(url_to_search)
            for i in data["edges"]:
                print("----------------")
                print(i["end"])
                print("relation:", i["rel"])
                print(i["sources"])
                print(i["start"])
                print("weight:", i["weight"])

    def IsA(self, term, limit=100):

        term = term.lower()

        if " " in term:
            term = term.replace(" ", "_")
        url_to_search = (
            self.web_api
            + "?start=/c/"
            + self.lan
            + "/"
            + term
            + "&?offset=0&limit=="
            + str(limit)
            + "&rel=/r/IsA"
        )
        data = requests.get(url_to_search).json()

        relations = []

        for i in data["edges"]:
            s = i["start"]["label"]
            r = i["rel"]["label"]
            e = i["end"]["label"]
            relations.append((s, r, e))

        return relations

    def HasContext(self, term, limit=100):

        term = term.lower()

        if " " in term:
            term = term.replace(" ", "_")
        url_to_search = (
            self.web_api
            + "?start=/c/"
            + self.lan
            + "/"
            + term
            + "&?offset=0&limit=="
            + str(limit)
            + "&rel=/r/HasContext"
        )
        data = requests.get(url_to_search).json()

        relations = []

        for i in data["edges"]:
            s = i["start"]["label"]
            r = i["rel"]["label"]
            e = i["end"]["label"]
            relations.append((s, r, e))

        return relations

    def Synonym(self, term, limit=100):

        term = term.lower()

        if " " in term:
            term = term.replace(" ", "_")
        url_to_search = (
            self.web_api
            + "?start=/c/"
            + self.lan
            + "/"
            + term
            + "&?offset=0&limit=="
            + str(limit)
            + "&rel=/r/Synonym"
        )
        data = requests.get(url_to_search).json()

        relations = []

        for i in data["edges"]:
            s = i["start"]["label"]
            r = i["rel"]["label"]
            e = i["end"]["label"]
            relations.append((s, r, e))

        return relations


# if __name__ == '__main__':
# ex_api=ConceptNet('en')
# ex_api.lookup(term='bitch',verbose=True,limit=10)
# print(ex_api.IsA(term='bitch',limit=10))
# print(ex_api.HasContext(term='bitch',limit=10))
# print(ex_api.Synonym(term='bitch',limit=10))
