from typing import List, Tuple, Dict
import gym.spaces
from selenium.webdriver.common.keys import Keys

from miniwob.environment import MiniWoBEnvironment
from miniwob.action import (
    MiniWoBType,
    MiniWoBElementClickId,
    MiniWoBElementClickXpath,
    MiniWoBElementClickOption,
    MiniWoBMoveXpath,
)
from miniwob import MINIWOB_DIR, EXTRA_HTML_TASKS


class MiniWoBHTMLEnv(gym.Env):
    def __init__(
        self,
        subdomain: str,
        headless: bool = False,
    ):
        self._env = MiniWoBEnvironment(subdomain)
        self.headless = headless
        self.task = None
        if self._env.subdomain not in [
            "click-collapsible-2",
            "click-tab-2",
            "click-tab-2-hard",
        ]:
            self._env.configure(
                num_instances=1,
                headless=self.headless,
                base_url=f"file://{MINIWOB_DIR}",
                seeds=[0],
                wait_ms=1000.0,
                refresh_freq=1,
            )

    def reset(
        self,
        seed: int = 0,
        record_screenshots: bool = False,
    ) -> str:
        """Forces stop and start all instances.

        Args:
            seed (int): Random seed to set the instance;
            record_screenshots (bool): Whether to record screenshots of the states.
        Returns:
            obs (str): The HTML of the initial state.
        """
        if self._env.subdomain in [
            "click-collapsible-2",
            "click-tab-2",
            "click-tab-2-hard",
        ]:
            self._env.configure(
                num_instances=1,
                headless=self.headless,
                base_url=f"file://{MINIWOB_DIR}",
                seeds=[0],
                wait_ms=1000.0,
                refresh_freq=1,
            )
        states = self._env.reset(seeds=[seed], record_screenshots=record_screenshots)
        assert len(states) == 1
        obs = self.state2html(states)
        self.task = states[0].utterance

        return obs

    def step(
        self,
        instruction: str,
    ) -> Tuple[str, int, bool, Dict]:
        """Converts the instruction into an action and calls the super step method."""
        instruction = instruction.split(" ")
        inst_type = instruction[0]
        inst_type = inst_type.lower()

        if inst_type == "type":
            characters = " ".join(instruction[1:])
            if self._env.subdomain == "book-flight":
                parts = characters.split("\\'")
                parts = [part.replace('"', "").replace("'", "") for part in parts]
                characters = "'".join(parts)
            else:
                characters = characters.replace('"', "").replace("'", "")
            action = MiniWoBType(characters)
        elif inst_type == "clickid":
            element_id = " ".join(instruction[1:])
            action = MiniWoBElementClickId(element_id)
        elif inst_type == "press":
            key_type = instruction[1].lower()
            if key_type == "enter":
                action = MiniWoBType("\n")
            elif key_type == "space":
                action = MiniWoBType(" ")
            elif key_type == "arrowleft":
                action = MiniWoBType(Keys.LEFT)
            elif key_type == "arrowright":
                action = MiniWoBType(Keys.RIGHT)
            elif key_type == "backspace":
                action = MiniWoBType(Keys.BACKSPACE)
            elif key_type == "arrowup":
                action = MiniWoBType(Keys.UP)
            elif key_type == "arrowdown":
                action = MiniWoBType(Keys.DOWN)
            elif key_type in ["command+a", "command+c", "command+v"]:
                action = MiniWoBType(key_type)
            else:
                raise ValueError("Invalid instruction")
        elif inst_type == "movemouse":
            xpath = " ".join(instruction[1:])
            action = MiniWoBMoveXpath(xpath)
        elif inst_type == "clickxpath":
            xpath = " ".join(instruction[1:])
            action = MiniWoBElementClickXpath(xpath)
        elif inst_type == "clickoption":
            xpath = " ".join(instruction[1:])
            action = MiniWoBElementClickOption(xpath)
        else:
            raise ValueError("Invalid instruction")

        states, rewards, dones, infos = self._env.step([action])
        obs = self.state2html(states)

        return obs, rewards[0], dones[0], infos["n"][0]

    def get_task(self):
        return self.task

    def close(self):
        self._env.close()

    def state2html(self, states: List) -> str:
        if states[0] is not None:
            obs = states[0].html_body
            if self._env.subdomain in EXTRA_HTML_TASKS:
                obs += states[0].html_extra
        else:
            obs = None

        return obs


if __name__ == "__main__":
    env = MiniWoBHTMLEnv("book-flight")
    # env = MiniWoBHTMLEnv("choose-date")
    # env = MiniWoBHTMLEnv("choose-list")
    # env = MiniWoBHTMLEnv("click-button-sequence")
    # env = MiniWoBHTMLEnv("click-button")
    # env = MiniWoBHTMLEnv("click-checkboxes-large")
    # env = MiniWoBHTMLEnv("click-checkboxes-soft")
    # env = MiniWoBHTMLEnv("click-checkboxes-transfer")
    # env = MiniWoBHTMLEnv("click-checkboxes")
    # env = MiniWoBHTMLEnv("click-collapsible-2")
    # env = MiniWoBHTMLEnv("click-collapsible")
    # env = MiniWoBHTMLEnv("click-color")
    # env = MiniWoBHTMLEnv("click-dialog-2")
    # env = MiniWoBHTMLEnv("click-dialog")
    # env = MiniWoBHTMLEnv("click-link")
    # env = MiniWoBHTMLEnv("click-menu")
    # env = MiniWoBHTMLEnv("click-option")
    # env = MiniWoBHTMLEnv("click-pie")
    # env = MiniWoBHTMLEnv("click-scroll-list")
    # env = MiniWoBHTMLEnv("click-shades")
    # env = MiniWoBHTMLEnv("click-shape")
    # env = MiniWoBHTMLEnv("click-tab-2-hard")
    # env = MiniWoBHTMLEnv("click-tab-2")
    # env = MiniWoBHTMLEnv("click-tab")
    # env = MiniWoBHTMLEnv("click-test-2")
    # env = MiniWoBHTMLEnv("click-test")
    # env = MiniWoBHTMLEnv("click-widget")
    # env = MiniWoBHTMLEnv("copy-paste-2")
    # env = MiniWoBHTMLEnv("copy-paste")
    # env = MiniWoBHTMLEnv("count-shape")
    # env = MiniWoBHTMLEnv("email-inbox-forward-nl-turk")
    # env = MiniWoBHTMLEnv("email-inbox-forward-nl")
    # env = MiniWoBHTMLEnv("email-inbox-nl-turk")
    # env = MiniWoBHTMLEnv("email-inbox")
    # env = MiniWoBHTMLEnv("enter-date")
    # env = MiniWoBHTMLEnv("enter-password")
    # env = MiniWoBHTMLEnv("enter-text-dynamic")
    # env = MiniWoBHTMLEnv("enter-text")
    # env = MiniWoBHTMLEnv("enter-time")
    # env = MiniWoBHTMLEnv("find-word")
    # env = MiniWoBHTMLEnv("focus-text-2")
    # env = MiniWoBHTMLEnv("focus-text")
    # env = MiniWoBHTMLEnv("grid-coordinate")
    # env = MiniWoBHTMLEnv("guess-number")
    # env = MiniWoBHTMLEnv("identify-shape")
    # env = MiniWoBHTMLEnv("login-user-popup")
    # env = MiniWoBHTMLEnv("login-user")
    # env = MiniWoBHTMLEnv("multi-layouts")
    # env = MiniWoBHTMLEnv("multi-orderings")
    # env = MiniWoBHTMLEnv("navigate-tree")
    # env = MiniWoBHTMLEnv("read-table")
    # env = MiniWoBHTMLEnv("search-engine")
    # env = MiniWoBHTMLEnv("simple-algebra")
    # env = MiniWoBHTMLEnv("simple-arithmetic")
    # env = MiniWoBHTMLEnv("social-media-all")
    # env = MiniWoBHTMLEnv("social-media-some")
    # env = MiniWoBHTMLEnv("social-media")
    # env = MiniWoBHTMLEnv("terminal")
    # env = MiniWoBHTMLEnv("text-transform")
    # env = MiniWoBHTMLEnv("tic-tac-toe")
    # env = MiniWoBHTMLEnv("unicode-test")
    # env = MiniWoBHTMLEnv("use-autocomplete")
    # env = MiniWoBHTMLEnv("use-spinner")

    for i in range(1, 200):
        se = i * 10000
        print("seed:", se)
        ob = env.reset(seed=se)
        print("task:", env.get_task())
        print("obs:")
        print(ob)
        done = False
        while not done:
            a = input("action: ")
            ob, reward, done, info = env.step(a)
            print("obs:")
            print(ob)
            print(reward, done, info)
    env.close()
