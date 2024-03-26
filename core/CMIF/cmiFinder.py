from abc import ABC, abstractmethod

from core.CMIF.utils import seed_checker


class Checker(ABC):

    @abstractmethod
    def consistency_checker(self, data_dir: str, dest_dir: str):
        """Abstract method for checking message condition pair consistency"""
        raise NotImplementedError


class CmiFinder(Checker):

    def consistency_checker(self, data_dir: str, dest_dir: str):
        """
        Function: To Find Consistency of Conditional Message Pair
        Input data should be in .jsonl format
        Destination directory will have a .jsonl file with only Consistent Conditional Message Pair
        """
        seed_checker.predict(DATA_DIR=data_dir, DEST_DIR=dest_dir)
