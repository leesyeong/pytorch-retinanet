import os
import sys
sys.path.append('ShipDetectUtility')

import glob
import json
import shutil
import os.path as osp
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

def GetFileName(path: str) -> str:
    """
    Extracts the file name from a path.

    Args:
        path (str): The file path.

    Returns:
        str: The file name.
    """
    return osp.splitext(osp.basename(path))[0]

def GetFileExtension(path: str) -> str:
    """
    Extracts the file extension from a path.

    Args:
        path (str): The file path.

    Returns:
        str: The file extension.
    """
    return osp.splitext(osp.basename(path))[1]

def GetFileDir(path: str) -> str:
    """
    Extracts the directory from a file path.

    Args:
        path (str): The file path.

    Returns:
        str: The directory path.
    """
    return osp.dirname(path)

def GetFiles(rootPath: str, filename: str = '', extension: str = '', isRecursive: bool = False) -> List[str]:
    """
    Retrieves a list of files from a specified directory.

    Args:
        rootPath (str): The directory path to start the search from.
        filename (str, optional): Filename filter. Defaults to ''.
        extension (str, optional): File extension filter. Defaults to ''.
        isRecursive (bool, optional): Whether to search recursively. Defaults to False.

    Returns:
        List[str]: A list of file paths.
    """
    results = []

    def search_files(directory):
        nonlocal results
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(extension):
                    results.append(osp.join(root, file))

    if isRecursive:
        search_files(rootPath)
    else:
        results = glob.glob(rootPath + '/*' + extension)

    filtered = []
    if filename != '':
        for result in results:
            if filename in osp.basename(result):
                filtered.append(result)
        results = filtered

    return results

def Xml2Dict(xmlfile: str) -> Union[dict, None]:
    """
    Converts an XML file to a dictionary.

    Args:
        xmlfile (str): The XML file path.

    Returns:
        Union[dict, None]: The converted dictionary or None if conversion fails.
    """
    try:
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        result = parse_element(root)
        return result
    except Exception as e:
        # stdout disable
        #print(f"Xml To Dict Error : {e}")
        return None

def Json2Dict(jsonfile: str) -> Union[dict, None]:
    """
    Converts a JSON file to a dictionary.

    Args:
        jsonfile (str): The JSON file path.

    Returns:
        Union[dict, None]: The converted dictionary or None if conversion fails.
    """
    try:
        with open(jsonfile, 'r') as f:
            result = json.load(f)

            if result is None or len(result) == 0 or type(result) is not dict:
                return None

            return result
    except Exception as e:
        # stdout disable
        # print(f"Json To Dict Error : {e}")
        return None

def parse_element(element) -> dict:
    """
    Converts an XML element to a dictionary.

    Args:
        element: The XML element.

    Returns:
        dict: The converted dictionary.
    """
    result = dict(element.attrib)

    if element.text and element.text.strip():
        result['text'] = element.text.strip()

    for child in element:
        child_data = parse_element(child)
        result.setdefault(child.tag, []).append(child_data)

    return result

def Pairing(pair1s: List[str], pair2s: List[str]) -> List[Tuple[str, str]]:
    """
    Pairs two lists of files.

    Args:
        pair1s (List[str]): The first list of files.
        pair2s (List[str]): The second list of files.

    Returns:
        List[Tuple[str, str]]: A list of paired file paths.
    """
    pair2s_dict = {GetFileName(pair2): pair2 for pair2 in pair2s}
    pairs = [(pair1, pair2s_dict[GetFileName(pair1)]) for pair1 in pair1s if GetFileName(pair1) in pair2s_dict]

    return pairs

def Remove(filepath: str) -> None:
    """
    Removes a file or directory.

    Args:
        filepath (str): The file or directory path.

    Raises:
        FileNotFoundError: If the file or directory does not exist.
    """
    if osp.exists(filepath):
        if osp.isdir(filepath):
            shutil.rmtree(filepath)
        else:
            os.remove(filepath)
    else:
        raise FileNotFoundError

def RemoveFiles(files: List[str]) -> None:
    """
    Removes multiple files.

    Args:
        files (List[str]): A list of file paths to remove.
    """
    for file in files:
        if osp.isfile(file):
            os.remove(file)
        else:
            print(f"File Not Found : {file}")

def RemoveDir(dir: str) -> None:
    """
    Removes a directory.

    Args:
        dir (str): The directory path to remove.
    """
    if osp.isdir(dir):
        os.rmdir(dir)
    else:
        print(f"Directory Not Found : {dir}")

def Exists(path: str) -> bool:
    """
    Checks if a file or directory exists.

    Args:
        path (str): The file or directory path.

    Returns:
        bool: True if the file or directory exists, False otherwise.
    """
    return osp.exists(path)
