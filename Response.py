import sys

sys.path.append("../src/")
import scipy.io
import numpy as np
import toolbox as tb
import re

PATTERN = "^(?P<home>.*)\/sub_(?P<subject>\d*)_exp(?P<experiment>\d)_session(?P<session>\d)_cat(?P<cat>\d)_img(?P<img>\d).(?P<ext>.*)"


class Response:

    def __init__(self, file):
        d = self.loadmat_(file)
        self.__dict__ = d["data"]
        self.filename = file
        self.parse_filename()
        self.fields = list(d["data"].keys())
        self.subject = self.filename.split("_")[1]
        return None

    def parse_filename(self):
        filename = self.filename
        self.fileinfo = re.match(PATTERN, filename).groupdict()

    def loadmat_(self, filename):
        """Improved loadmat (replacement for scipy.io.loadmat)
        Ensures correct loading of python dictionaries from mat files.

        Inspired by: https://stackoverflow.com/a/29126361/572908
        """

        def _has_struct(elem):
            """Determine if elem is an array
            and if first array item is a struct
            """
            return (
                isinstance(elem, np.ndarray)
                and (elem.size > 0)
                and isinstance(elem[0], scipy.io.matlab.mio5_params.mat_struct)
            )

        def _check_keys(d):
            """checks if entries in dictionary are mat-objects. If yes
            todict is called to change them to nested dictionaries
            """
            for key in d:
                elem = d[key]
                if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                    d[key] = _todict(elem)
                elif _has_struct(elem):
                    d[key] = _tolist(elem)
            return d

        def _todict(matobj):
            """A recursive function which constructs from
            matobjects nested dictionaries
            """
            d = {}
            for strg in matobj._fieldnames:
                elem = matobj.__dict__[strg]
                if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                    d[strg] = _todict(elem)
                elif _has_struct(elem):
                    d[strg] = _tolist(elem)
                else:
                    d[strg] = elem
            return d

        def _tolist(ndarray):
            """A recursive function which constructs lists from cellarrays
            (which are loaded as numpy ndarrays), recursing into the
            elements if they contain matobjects.
            """
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, scipy.io.matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif _has_struct(sub_elem):
                    elem_list.append(_tolist(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return elem_list

        data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return _check_keys(data)

    def get_tested_pairs(self, transform=True, return_rt=True,all_pairs=True):
        tested_pairs = self.testedPairs
        if all_pairs: 
            uniques = np.unique(tested_pairs)
            tested_pairs = []
            for i in range(len(uniques)):
                for j in range(i+1,len(uniques)):
                    tested_pairs.append([uniques[i],uniques[j]])
        _pair_coords = [
            (
                [self.xGrid[pair[0] - 1], self.yGrid[pair[0] - 1]],
                [self.xGrid[pair[1] - 1], self.yGrid[pair[1] - 1]],
            )
            for pair in tested_pairs
        ]

        if transform:
            transform = lambda x: [
                tb.transform_coord_system(x[0]),
                tb.transform_coord_system(x[1]),
            ]

            pair_coords = [transform(coord) for coord in _pair_coords]
        else:
            pair_coords = _pair_coords

        out = pair_coords

        if return_rt:
            assert len(self.testedPairs) == len(self.reactionTime)
            out = (pair_coords,self.reactionTime)
        
        return pair_coords

