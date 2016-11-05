class Counter(dict):
    '''
    this class is used to do basic counting and smoothing for events

    It is designed via dictionary
    '''

    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def argMax(self):

        # return key with highest value
        if len(self.keys()) == 0:
            return None
        items = self.items()  # items=[(key,value)...]
        values = [i[1] for i in items]
        keys = [i[0] for i in items]
        max_idx = values.index(max(values))
        res = list(items)[max_idx][0]  # find the key
        return res

    def totalCount(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())

    def normalize(self):
        total = float(self.totalCount())
        if total == 0: return False
        for k in self.keys():
            self[k] = self[k] / total

    def copy(self):

        return Counter(dict.copy(self))
