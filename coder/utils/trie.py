
class TrieNode:
    def __init__(self, key=None):
        self.key = key
        self.children = {}
        self.score = 0
        self.is_valid = True
        self.is_end_of_sequence = False
    
    def get_children(self):
        return list(self.children.values())
    
    def set_score(self, score):
        self.score = score

    def remove_child(self, key):
        self.children.pop(key)
        if len(self.children) == 0:
            self.is_valid = True

        

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, sequence):
        node = self.root
        for key in sequence:
            if key not in node.children:
                node.children[key] = TrieNode(key)
            node = node.children[key]
        node.is_end_of_sequence = True

    def get_all_valid_sequences(self):
        sequences = []
        def _dfs(node, seq, score=0, depth=1):
            seq.append(node.key)
            score += node.score
            if not node.is_valid:
                return
            if node.is_end_of_sequence:
                sequences.append((seq, score/depth))
                return
            for child in node.get_children():
                _dfs(child, seq.copy(), score, depth+1)
        
        for node in self.root.get_children():
            _dfs(node, [])
        return sequences



