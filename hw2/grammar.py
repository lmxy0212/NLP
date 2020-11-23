"""
COMS W4705 - Natural Language Processing - Fall B 2020
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba

Completed by Manxeuying Li
"""

import sys
from collections import defaultdict
from math import fsum, isclose


class Pcfg(object):
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file):
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None
        self.read_rules(grammar_file)

    def read_rules(self, grammar_file):

        for line in grammar_file:
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line:
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else:
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()

    def parse_rule(self, rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";", 1)
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
        for lhs, rule in self.lhs_to_rules.items():
            #lhs should always be upper
            if not lhs.isupper():
                return False
            prob_lst = []
            for curr in rule:
                #total length should be 3 (lhs, rhs, prob)
                #grammar in tuple
                #prob should be float
                #length 2 on rhs + all upper
                #length 1 on rhs + lower
                if (len(curr) != 3)\
                 or not(isinstance(curr[1], tuple))\
                 or not (isinstance(curr[2], (int, float)))\
                 or not ((len(curr[1]) == 2 and curr[1][0].isupper() and curr[1][1].isupper()))\
                 or not (len(curr[1]) == 1 and not curr[1][0].isupper()):
                 return False
                else:
                 prob_lst.append(curr[2])

            #prob sum to 1
            if not isclose(fsum(prob_lst), 1.): 
                return False

        return True


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as grammar_file:
        grammar = Pcfg(grammar_file)

    if grammar.verify_grammar():
        print("It is a valid grammar")
    else:
        print("error: Not a valid grammar")
