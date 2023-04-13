"""Custom git-theta merge tool."""

import argparse
import asyncio
import functools
import itertools
import logging
import os
import sys
import tempfile
from typing import Any, Dict, FrozenSet, List, Optional, Union

from prompt_toolkit import PromptSession, print_formatted_text, prompt
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.validation import ValidationError, Validator

from git_theta import async_utils, merges, metadata
from git_theta.utils import TEXT_STYLE, DiffState, EnvVarConstants, NoResult, Trie

logging.basicConfig(
    level=logging.DEBUG,
    # Log to a file for clean/smudge as they don't appear on the console when called via git.
    filename=os.path.join(tempfile.gettempdir(), "git-theta.log"),
    format="git-theta-merge: [%(asctime)s] %(levelname)s - %(message)s",
)


def infer_state(
    ancestor: Optional[metadata.ParamMetadata],
    current: Optional[metadata.ParamMetadata],
    other: Optional[metadata.ParamMetadata],
) -> DiffState:
    """Convert differences between each branch into a semantic difference."""
    if ancestor == current == other:
        return DiffState.EQUAL
    if ancestor == other and current != ancestor:
        if ancestor is None:
            return DiffState.ADDED_A
        if current is None:
            return DiffState.DELETED_A
        return DiffState.CHANGED_A
    if ancestor == current and current != other:
        if ancestor is None:
            return DiffState.ADDED_B
        if current is None:
            return DiffState.DELETED_B
        return DiffState.CHANGED_B
    if ancestor is None:
        return DiffState.DELETED_B
    return DiffState.CHANGED_BOTH


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="git-theta merge program")
    parser.add_argument("ancestor")  # %O
    parser.add_argument("current")  # %A
    parser.add_argument("other")  # %B
    parser.add_argument("path")  # %P
    args = parser.parse_args()
    return args


class CommandValidator(Validator):
    """Only allow valid commands."""

    def __init__(self, avail: Dict[str, Any], prefixes: Trie, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avail = avail
        self.prefixes = prefixes

    def validate(self, document):
        text = document.text
        # The `in` check is done with the dictionary of kb short cuts -> action
        # instead of the trie because it is O(1) % cost of hashing while an `in`
        # operation on a trie is O(n)
        if text and text not in self.avail:
            # The trie lets us validate that the user is typing something that
            # could be a value in the auto complete list.
            if self.prefixes.prefix(text):
                # Raise this error to stop the user from hitting enter but don't
                # show that anything is actually wrong.
                raise ValidationError(message="", cursor_position=len(text))
            raise ValidationError(
                message="This input is not an allowed action.",
                cursor_position=len(text),
            )


class FilteredAutoSuggestFromHistory(AutoSuggestFromHistory):
    """AutoSuggest that is aware of current allowable actions."""

    def __init__(self, *args, valid_suggestions, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_suggestions = valid_suggestions

    def get_suggestion(self, buffer, document):
        suggestion = super().get_suggestion(buffer, document)
        # Suggestions are populated from history, but make sure we don't
        # populate with a command that was valid in the past but not now.
        if suggestion:
            full_action = f"{document.text}{suggestion.text}"
            suggestion = suggestion if full_action in self.valid_suggestions else None
        return suggestion


def make_short_cuts(
    handlers: Dict[str, merges.Merge], reserved: FrozenSet[str] = frozenset({"q"})
):
    """Convert the loaded plugin to a map from action triggering string to handler.

    Note:
        Each handler can request a specific string to use for selection via the
        SHORT_CUT class attribute, if that string is already in-use then the
        next value from an incrementing series of numbers is used as the
        section string.

        Plug-ins are processed alphabetically based on the name used to register
        them. If these names are eventually abused to ensure some plugin gets a
        specific action string (i.e. it is named AAAAAActualName) we may want to
        add a display name attribute.
    """
    short_cuts = {}
    action = 1
    for _, handler in sorted(handlers.items()):
        # Warn if there is a plug-in using a reserved action keyword.
        # A = short_cut is reserved
        # B = short_cut is free
        if handler.SHORT_CUT in reserved:
            # Triggers for [A ∧ B, A ∧ ¬B]
            logging.warning(
                f"Merge Plug-in {handler.NAME} requested short-cut"
                f" {handler.SHORT_CUT} which is reserved."
            )
        # If their requested shortcut is available, give it to them.
        elif handler.SHORT_CUT not in short_cuts:
            # elif makes sure this only happens if the short cut is not in use and
            # it is not in the reserved set.
            # Triggers for [¬A ∧ B]
            short_cuts[handler.SHORT_CUT] = handler
            continue
        # Otherwise give them the next number.
        # Triggers for [A ∧ B, A ∧ ¬B, ¬A ∧ ¬B]
        short_cuts[action] = handler
        action += 1
    return short_cuts


def filter_actions(
    state: DiffState, actions: Dict[str, merges.Merge]
) -> Dict[str, merges.Merge]:
    """Filter out actions that are not applicable in this state."""
    return {
        kb: action
        for kb, action in actions.items()
        if state not in action.INACTIVE_STATES
    }


def build_menu(
    actions: Dict[str, Union[str, merges.Merge]], indent: int = 2
) -> List[str]:
    """Convert the mapping of strings to actions into a menu."""
    menu = []
    # Format action strings so they all end at the same place.
    longest_kb = max(len(kb) for kb in actions)
    for kb, action in actions.items():
        menu.append(f"{kb:>{longest_kb}})  {action}")
    # Add a small indent to each action.
    menu = [f"{'':>{indent}}{m}" for m in menu]
    return menu


def manual_merge(args):
    logging.info(f"Writing model weights from {args.path} for manual merging.")

    # TODO: Update smudge API to make it easier to call programatically.
    async def load(name, path):
        with open(path, "rb") as f:
            raw_weights = (
                await async_utils.subprocess_run(
                    ["git-theta-filter", "smudge", args.path],
                    f.read(),
                    capture_output=True,
                )
            ).stdout
        return name, raw_weights

    checkpoints = {
        "ours": args.current,
        "theirs": args.other,
        "ancestor": args.ancestor,
    }

    checkpoints = async_utils.run(async_utils.run_map(checkpoints, load))
    for name, raw_weights in checkpoints.items():
        with open(f"{name}.ckpt", "wb") as wf:
            logging.info(f"Saving {name} model to {name}.ckpt")
            wf.write(raw_weights)
    logging.info(
        "Manual Merging: Combine checkpoints as you wish, save the "
        f"result to {args.path} and continue the merge."
    )
    sys.exit(1)


def merge(args):
    """git-theta checkpoint aware merging."""
    print_formatted_text(
        HTML(f"<b>Fixing Merge Conflicts in {TEXT_STYLE.format_model(args.path)}</b>")
    )
    logging.debug(f"Running merge driver on {args.path}")
    # Load the `cleaned` metadata file for the ancestor commit.
    ancestor = metadata.Metadata.from_file(args.ancestor)
    ancestor = ancestor.flatten()
    # Load the `cleaned` metadata file for commit on our branch.
    current = metadata.Metadata.from_file(args.current)
    current = current.flatten()
    # Load the `cleaned` metadata file for commit on the other branch.
    other = metadata.Metadata.from_file(args.other)
    other = other.flatten()

    # Collect the list of all parameter names. NB: Names are collected from all
    # models as they may have been deleted/added on different branches.
    all_params = sorted(
        list(set(itertools.chain(ancestor.keys(), current.keys(), other.keys())))
    )
    # Load all merge handlers and assign each one a kb shortcut.
    handlers = merges.all_merge_handlers()
    short_cuts = make_short_cuts(handlers)

    # Trigger the context merge for summary of branches.
    handlers["context"]().merge()

    # A place to aggregate metadata for the merged model.
    merged_model = {}
    # A place to store loaded parameters to enable reuse/caching. Currently
    # modified in-place :(
    partial_current = {}
    partial_other = {}
    partial_ancestor = {}
    # Create a `prompt_toolkit` session so we can auto suggest action based on
    # history i.e. if they keep taking the other branches change, suggest that
    # for the next parameter.
    session = PromptSession()
    for param_name in all_params:
        # Get each the parameter metadata from each model.
        ancestor_param = ancestor.get(param_name)
        current_param = current.get(param_name)
        other_param = other.get(param_name)
        # The parameter name with / for scoping.
        name = "/".join(param_name)

        # What happened to the parameter across branches?
        state = infer_state(ancestor_param, current_param, other_param)

        # Changes that don't need human action to be resolved.
        # If the parameter is unchanged between all models just use it.
        if state is DiffState.EQUAL:
            logging.debug(
                f"Parameter {name} was unchanged by either branch. "
                "Keeping the value the same going forward."
            )
            merged_model[param_name] = ancestor_param
            continue
        if state is DiffState.DELETED_BOTH:
            logging.debug(
                f"Parameter {name} was deleted on both branches. "
                "Removing from the merged result."
            )
            continue

        # Enumerate the actions you can take based on the state of the params.
        available_actions = filter_actions(state, short_cuts)
        # Add a quit action.
        available_actions["q"] = "quit"
        # A Trie for prefix loop up of available action commands.
        available_prefixes = Trie.from_iterable(available_actions)

        # Build an action menu.
        text = [
            # What parameter we are working with and what happened to it.
            f"<b>{TEXT_STYLE.format_param(name)}</b>: {state.value}",
            "Actions:",
        ]
        # Add allowed actions.
        text.extend(build_menu(available_actions))
        # Input prompt.
        text.append("𝜃 ")
        text = HTML("\n".join(text))
        # An info box at the bottom of the cli, helps remind users of the context
        # they are working on.
        context = HTML(
            f'Merging parameter: <b><style bg="{TEXT_STYLE.param}">{name}</style></b>'
            f' in model <i><style bg="{TEXT_STYLE.model}">{args.path}</style></i>'
        )

        merged_parameter = NoResult
        while merged_parameter is NoResult:
            # User action selection.
            action = session.prompt(
                # Show the menu
                text,
                # Show the merge context
                bottom_toolbar=context,
                # Validate that their input is either a valid action or is the
                # prefix of a valid action. This lets us alert early if their input
                # is not in the autocomplete menu.
                validator=CommandValidator(available_actions, available_prefixes),
                # Suggest completions they have used before, accept with ->
                auto_suggest=FilteredAutoSuggestFromHistory(
                    valid_suggestions=available_actions
                ),
                # Give them a list of autocomplete actions based on the valid actions.
                completer=WordCompleter(available_actions),
                # Pop up the completion list as they type.
                complete_while_typing=True,
            )
            action = action.strip()
            logging.debug(f"User Input: {action}")
            if action == "q":
                logging.debug(
                    "User quit the merge tool. Leaving merge files as they are."
                )
                sys.exit(1)

            # Collect any arguments that the merge action requires.
            action_arguments = {}
            for merge_argument in available_actions[action].merge_arguments():
                param_text = HTML(
                    "\n".join(
                        [
                            merge_argument.description,
                            f"<b>{TEXT_STYLE.format_argument(merge_argument.name)}</b>: ",
                        ]
                    )
                )
                validator = Validator.from_callable(merge_argument.validator)
                argument_value = prompt(param_text, validator=validator)
                # TODO: Add backend validation of the argument value
                action_arguments[merge_argument.name] = merge_argument.type(
                    argument_value
                )

            # Dispatch based on action
            # TODO: When should these objects be initialized?
            # TODO: How should we configure these actions?
            # TODO: Move to a verb object compositional approach?
            # TODO: Move to async for actions?
            merged_parameter = available_actions[action]()(
                param_name,
                current_param,
                other_param,
                ancestor_param,
                current,
                other,
                ancestor,
                # TODO: Update API so these don't need to be in-place updates?
                partial_current,
                partial_other,
                partial_ancestor,
                # The path to where the model actually lives.
                args.path,
                # Merge action-specific parameters
                **action_arguments,
            )
        # Some of the actions result in deleting a parameter (it has a value of
        # None) so if we see that, don't add this parameter name to the merged
        # model.
        if merged_parameter is None:
            continue
        merged_model[param_name] = merged_parameter

    merged_model = metadata.Metadata(**merged_model).unflatten()
    # Save merged_model to args.current %A
    merged_model.write(args.current)
    # Exit with 0 to signal the merge was good.
    return 0


def main():
    args = parse_args()
    if EnvVarConstants.MANUAL_MERGE:
        manual_merge(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
