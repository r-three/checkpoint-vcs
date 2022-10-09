import git
import os
import json
import logging

def get_git_repo():
    """
    Create a git.Repo object for this repository

    Returns
    -------
    git.Repo
        Repo object for the current git repository
    """
    return git.Repo(os.getcwd(), search_parent_directories=True)

def create_git_ml(repo):
    """
    If not already created, create $git_root/.git_ml and return path

    Parameters
    ----------
    git_root : str
        path to git repository's root directory

    Returns
    -------
    str
        path to $git_root/.git_ml directory
    """
    git_ml = os.path.join(repo.working_dir, ".git_ml")
    if not os.path.exists(git_ml):
        logging.debug(f"Creating git ml directory {git_ml}")
        os.makedirs(git_ml)
    return git_ml

def create_git_ml_model_dir(repo, model_path):
    """
    If not already created, create directory under $git_root/.git_ml/ to store a model and return path

    Parameters
    ----------
    repo : git.Repo
        Repo object for the current git repository

    model_path : str
        path to model file being saved

    Returns
    -------
    str
        path to $git_root/.git_ml/$model_name directory
    """
    git_ml = create_git_ml(repo)
    model_file = os.path.basename(model_path)
    git_ml_model = os.path.join(git_ml, os.path.splitext(model_file)[0])

    if not os.path.exists(git_ml_model):
        logging.debug(f"Creating model directory {git_ml_model}")
        os.makedirs(git_ml_model)
    return git_ml_model

def load_tracked_file(f):
    """
    Load tracked file
    TODO: currently implemented for json but should really be Pytorch/TF checkpoints

    Parameters
    ----------
    f : str
        path to file tracked by git-cml filter

    Returns
    -------
    dict
        contents of file

    """
    logging.debug(f"Loading tracked file {f}")
    with open(f, "r") as f:
        return json.load(f)

def write_tracked_file(f, param):
    """
    Dump param into a file
    TODO: currently dumps as json but should really be format designed for storing tensors on disk

    Parameters
    ----------
    f : str
        path to output file
    param : list or scalar
        param value to dump to file

    """
    logging.debug(f"Dumping param to {f}")
    with open(f, "w") as f:
        json.dump(param, f)

def add_file(f, repo):
    """
    Add file to git staging area

    Parameters
    ----------
    f : str
        path to file
    repo : git.Repo
        Repo object for current git repository
    """
    logging.debug(f"Adding {f} to staging area")
    repo.git.add(f)
