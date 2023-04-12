import argparse
import logging
from pathlib import Path

from agent import Agent
from env import MiniWoBHTMLEnv
from exemplar_lib import ExemplarLibrary
from utils import save_result

logging.basicConfig(level=logging.INFO)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--env_name", type=str, default="click-button")
    parser.add_argument("--api_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--no_filter", action="store_true", default=False)
    parser.add_argument("--exemplar_decomposition", action="store_true", default=False)
    parser.add_argument(
        "--reformat_input", type=str, default=None, choices=["obs", "task", "obs_task"]
    )
    parser.add_argument("--task_as_reformation", action="store_true", default=False)
    parser.add_argument("--no_multi_task", action="store_true", default=False)
    parser.add_argument("--plugin_name", type=str, default=None, choices=["minimax"])
    parser.add_argument("--ignore_failure", action="store_true", default=False)
    parser.add_argument("--init_db", action="store_true", default=False)
    parser.add_argument("--top_k", type=int, default=1)
    return parser


def run(args):
    assert args.api_name == "gpt-3.5-turbo"
    if not args.no_multi_task:
        db = ExemplarLibrary(args)
    env = MiniWoBHTMLEnv(
        subdomain=args.env_name.replace("-ablation", ""), headless=args.headless
    )

    success = 0
    for i in range(args.num_episodes):
        reward = 0
        done = False
        current_seed = args.seed + i

        state = env.reset(seed=current_seed)
        task = env.get_task()
        if args.no_multi_task:
            if args.env_name == "click-tab-2-hard":
                exemplar_name = "click-tab-2"
            else:
                exemplar_name = args.env_name
        else:
            query = "Task: " + task + "\nState:\n" + state
            exemplar_name = db.retrieve_exemplar_name(query)

        decompose_str = "decompose_" if args.exemplar_decomposition else ""
        reformat_str = "" if args.reformat_input is None else "reformat_"
        filter_str = "" if args.no_filter else "filter_"
        match_str = "" if exemplar_name == args.env_name else f"_{exemplar_name}"
        log_path = Path(
            f"results/{args.api_name}/{args.env_name}/{decompose_str}{reformat_str}{filter_str}seed{current_seed}{match_str}.txt"
        )

        try:
            agent = Agent(
                args=args,
                exemplar_name=exemplar_name,
                log_path=log_path,
            )
        except:
            logging.error("Agent initialization failed")
            logging.info(f"Seed {current_seed}: Fail")
            save_result("FAIL", log_path)
            continue

        if args.exemplar_decomposition and "SubtaskReformation" in agent.prompts:
            agent.reformat_subtasks(task)

        for _ in range(args.max_steps):
            # Observation filtering and reformation.
            if args.task_as_reformation:
                reformation = task
            else:
                if args.reformat_input == "task":
                    obs = task
                elif args.no_filter:
                    obs = state
                else:
                    obs = agent.obs_filter(state)

                if args.reformat_input is None:
                    reformation = obs
                elif args.reformat_input == "obs_task":
                    reformation = agent.task_reformation(obs, task)
                elif args.reformat_input == "task":
                    reformation = agent.task_reformation(task)
                else:
                    reformation = agent.task_reformation(obs)

            # Planning based on reformation.
            plan = agent.plan(reformation)
            for step, action in enumerate(plan):
                try:
                    state, reward, done, info = env.step(action)
                except:
                    info = {"action_fail": True}
                if not info["action_fail"]:
                    logging.info(f"Action executed: {action}")
                elif not args.ignore_failure:
                    logging.info(f"Invalid action: {action}")
                    reward = 0
                    done = True

                if done:
                    break

            if args.exemplar_decomposition:
                done = done or agent.update_exemplars()

            if done:
                break

        if reward > 0:
            success += 1
            logging.info(f"Seed {current_seed}: Success")
            save_result("SUCCESS", log_path)
        else:
            if args.env_name == "tic-tac-toe" and reward == -0.5:
                logging.info(f"Seed {current_seed}: Draw")
                save_result("DRAW", log_path)
            else:
                logging.info(f"Seed {current_seed}: Fail")
                save_result("FAIL", log_path)

    logging.info(f"Success rate: {success} / {args.num_episodes}")

    env.close()


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
