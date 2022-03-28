import string

def init_skip_symbol():
    global skip_symbol
    global erase_symbol
    skip_symbol = f"0123456789{string.ascii_letters}"
    erase_symbol = "\n,./;'[]\<>?:\"{}|_+(*&^%$#@!)（），。．？￼」 ⋯「！"
    
def split_to_word(sentense:str,word_len : int):
    if(len(sentense) == 0 or len(sentense) < word_len):
        return []
    temp_seq = ""
    word_container = []
    word_sequence = []
    
    def push_word(force_update : bool = False):
        cur_len = len(word_sequence) 
        if(cur_len == 0 or (not force_update and cur_len < word_len)): 
            return
        word_container.append("".join(word_sequence))
        word_sequence.clear()

    for each_char in sentense:
        if each_char in skip_symbol:
            temp_seq += each_char
            continue
        if len(temp_seq) > 0:
            word_sequence.append(temp_seq)
            temp_seq = ""
            push_word()

        word_sequence.append(each_char)
        push_word()

    if(len(temp_seq) > 0):
        word_sequence.append(temp_seq)
    push_word(True)

    return word_container

def extract_gram(sentense : str,word_len : int):
    last_char = ""
    words = []
    for char_pos in range(0, len(sentense)):
        if last_char in skip_symbol and sentense[char_pos] in skip_symbol:
            continue
        words = [*words,*list(filter(lambda w: w not in words,split_to_word(sentense[char_pos:], word_len)))]
        last_char = sentense[char_pos]
    return words
init_skip_symbol()