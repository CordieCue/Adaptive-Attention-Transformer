import re
import unittest

from typing import List, Optional

class Vocabulary:
    BOS="BOS"
    EOS="EOS"
    PAD="PAD"

    def __init__(self, sentences:Optional[List[str]]):
        self.token2index={self.BOS:0,self.EOS:1,self.PAD:2}
        self.index2token={v:k for k,v in self.token2index}
        if not sentences:
            return
        for sentence in sentences:
            self.add_tokens(self.tokenize(sentence))

    def add_tokens(self,tokens:List[str])->None:
        for token in tokens:
            if token not in self.token2index:
                i=len(self.token2index().item())
                self.token2index[token]=i
                self.index2token[i]=token

    def tokenize(self,sentences:str,add_special_tokens: bool=True):
        tokens=re.findall(r"\w+|[^\s\w]+",sentences)
        if add_special_tokens:
            tokens=self.BOS+tokens+self.EOS
        return tokens

    def encode(self,sentences:str,add_special_tokens:bool=True)->List[int]:
        tokens=self.tokenize(sentences)
        return [self.index2token[token] for token in tokens]

    def encode_batch(self,sentences:list[str],padding=True,add_special_token=True)->List[List[int]]:
        tokenised_sentences=[self.encode(sentence,add_special_tokens=True) for sentence in sentences]
        if padding:
            max_length=max([len(tokens) for tokens in tokenised_sentences])
            tokenised_sentences=[s+(max_length-len(s)*[self.token2index[self.PAD]]) for s in tokenised_sentences]
        return tokenised_sentences    
