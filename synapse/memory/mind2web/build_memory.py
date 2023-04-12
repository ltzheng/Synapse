import openai
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import json
from lxml import etree
import pickle

from synapse.envs.mind2web.env_utils import (
    load_json,
    get_target_obs_and_act,
    get_target_obs,
)


def get_top_k_obs(s: dict, top_k: int) -> tuple[str, str]:
    pos_candidates = s["pos_candidates"]
    pos_ids = [c["backend_node_id"] for c in pos_candidates]
    neg_candidates = s["neg_candidates"]
    neg_candidates = [c for c in neg_candidates if c["rank"] < top_k]
    neg_ids = [c["backend_node_id"] for c in neg_candidates]
    all_candidates = pos_ids + neg_ids
    obs = get_target_obs(etree.fromstring(s["cleaned_html"]), all_candidates)
    if len(s["pos_candidates"]) == 0:
        # Simplify the raw_html if pos_candidates is empty (not in the cleaned html)
        dom_tree = etree.fromstring(s["raw_html"])
        gt_element = dom_tree.xpath(f"//*[@data_pw_testid_buckeye='{s['action_uid']}']")
        element_id = gt_element[0].get("backend_node_id")
        raw_obs = get_target_obs(dom_tree, [element_id])
        # Find the start index of the target element using the element ID
        start_idx = raw_obs.find(f"id={element_id}")
        # Find the start tag for the target element
        start_tag_idx = raw_obs.rfind("<", 0, start_idx)
        end_tag_idx = raw_obs.find(">", start_idx)
        # Extract the tag name
        tag_name = raw_obs[start_tag_idx + 1 : end_tag_idx].split()[0]
        # Initialize count for open and close tags
        open_count = 0
        close_count = 0
        search_idx = start_tag_idx
        while True:
            # Find the next open or close tag of the same type
            next_open_tag = raw_obs.find(f"<{tag_name}", search_idx)
            next_close_tag = raw_obs.find(f"</{tag_name}>", search_idx)
            # No more tags found, break
            if next_open_tag == -1 and next_close_tag == -1:
                break
            # Decide whether the next tag is an open or close tag
            if next_open_tag != -1 and (
                next_open_tag < next_close_tag or next_close_tag == -1
            ):
                open_count += 1
                search_idx = raw_obs.find(">", next_open_tag) + 1
            else:
                close_count += 1
                search_idx = next_close_tag + len(f"</{tag_name}>")
            # If we've closed all open tags, break
            if open_count == close_count:
                break
        # Extract the target element
        target_element = raw_obs[start_tag_idx:search_idx]
        obs = obs.replace("</html>", f"{target_element} </html>")
    else:
        target_element = None

    return obs, target_element


def get_specifiers_from_sample(sample: dict) -> str:
    website = sample["website"]
    domain = sample["domain"]
    subdomain = sample["subdomain"]
    goal = sample["confirmed_task"]
    specifier = (
        f"Website: {website}\nDomain: {domain}\nSubdomain: {subdomain}\nTask: {goal}"
    )

    return specifier


def build_memory(memory_path: str, data_dir: str):
    top_k = 5
    score_path = "scores_all_data.pkl"
    with open(os.path.join(data_dir, score_path), "rb") as f:
        candidate_results = pickle.load(f)
    candidate_scores = candidate_results["scores"]
    candidate_ranks = candidate_results["ranks"]

    specifiers = []
    exemplars = []
    samples = load_json(data_dir, "train")
    for sample in samples:
        specifiers.append(get_specifiers_from_sample(sample))
        prev_obs = []
        prev_actions = []
        for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
            # add prediction scores and ranks to candidates
            sample_id = f"{sample['annotation_id']}_{s['action_uid']}"
            for candidates in [s["pos_candidates"], s["neg_candidates"]]:
                for candidate in candidates:
                    candidate_id = candidate["backend_node_id"]
                    candidate["score"] = candidate_scores[sample_id][candidate_id]
                    candidate["rank"] = candidate_ranks[sample_id][candidate_id]

            _, target_act = get_target_obs_and_act(s)
            target_obs, _ = get_top_k_obs(s, top_k)

            if len(prev_obs) > 0:
                prev_obs.append("obs: `" + target_obs + "`")
            else:
                query = f"Task: {sample['confirmed_task']}\nTrajectory:\n"
                prev_obs.append(query + "obs: `" + target_obs + "`")
            prev_actions.append("act: `" + target_act + "` (" + act_repr + ")")

        message = []
        for o, a in zip(prev_obs, prev_actions):
            message.append({"role": "user", "content": o})
            message.append({"role": "user", "content": a})
        exemplars.append(message)

    with open(os.path.join(memory_path, "exemplars.json"), "w") as f:
        json.dump(exemplars, f, indent=2)

    print(f"# of exemplars: {len(exemplars)}")

    # embed memory_keys into VectorDB
    openai.api_key = os.environ["OPENAI_API_KEY"]
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    metadatas = [{"name": i} for i in range(len(specifiers))]
    memory = FAISS.from_texts(
        texts=specifiers,
        embedding=embedding,
        metadatas=metadatas,
    )
    memory.save_local(memory_path)


def retrieve_exemplar_name(memory, query: str, top_k) -> list[str]:
    docs_and_similarities = memory.similarity_search_with_score(query, top_k)
    retrieved_exemplar_names = []
    scores = []
    for doc, score in docs_and_similarities:
        retrieved_exemplar_names.append(doc.metadata["name"])
        scores.append(score)

    return retrieved_exemplar_names, scores


def load_memory(memory_path):
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    memory = FAISS.load_local(memory_path, embedding)

    return memory
