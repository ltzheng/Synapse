import logging
import json

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from collections import Counter


EXEMPLAR_LIST = [
    "book-flight",
    "choose-date",
    "click-button-sequence",
    "click-button",
    "click-checkboxes-large",
    "click-checkboxes-soft",
    "click-collapsible-2",
    "click-collapsible",
    "click-color",
    "click-dialog-2",
    "click-dialog",
    "click-link",
    "click-menu",
    "click-pie",
    "click-scroll-list",
    "click-shades",
    "click-shape",
    "click-tab-2",
    "click-tab",
    "click-widget",
    "copy-paste-2",
    "count-shape",
    "email-inbox-nl-turk",
    "enter-date",
    "enter-password",
    "enter-text-dynamic",
    "enter-time",
    "find-word",
    "focus-text-2",
    "focus-text",
    "grid-coordinate",
    "guess-number",
    "identify-shape",
    "login-user-popup",
    "multi-layouts",
    "navigate-tree",
    "read-table",
    "search-engine",
    "simple-algebra",
    "social-media-all",
    "social-media-some",
    "social-media",
    "terminal",
    "text-transform",
    "tic-tac-toe",
    "use-autocomplete",
    "use-spinner",
]


class ExemplarLibrary:
    def __init__(
        self,
        args,
    ):
        embedding = OpenAIEmbeddings()
        if args.init_db:
            logging.info("Initializing FAISS")
            with open("exemplars.json", "r") as rf:
                exemplar_dict = json.load(rf)
                exemplar_names = []
                exemplar_descriptions = []
                for k, v in exemplar_dict.items():
                    if k in EXEMPLAR_LIST:
                        for query in v["Specifier"]:
                            exemplar_names.append(k)
                            exemplar_descriptions.append(query)
            self.db = FAISS.from_texts(
                texts=exemplar_descriptions,
                embedding=embedding,
                metadatas=[{"name": name} for name in exemplar_names],
            )
            self.db.save_local("exemplar_lib")
        else:
            logging.info("Loading FAISS")
            self.db = FAISS.load_local("exemplar_lib", embedding)

        self.retriever = self.db.as_retriever(search_kwargs={"k": args.top_k})
        logging.info(f"Number of embeddings: {len(self.db.index_to_docstore_id)}")

    def retrieve_exemplar_name(self, query: str) -> str:
        docs = self.retriever.get_relevant_documents(query)
        retrieved_exemplar_names = [doc.metadata["name"] for doc in docs]
        logging.info(f"Retrieved exemplars: {retrieved_exemplar_names}")
        data = Counter(retrieved_exemplar_names)
        retrieved_exemplar_name = data.most_common(1)[0][0]

        return retrieved_exemplar_name
