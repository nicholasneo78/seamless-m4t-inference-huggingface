from nltk import flatten
from num2words import num2words
from text2digits import text2digits
from hanziconv import HanziConv
import string
import re
import decimal
import logging

# Setup logging in a nice readable format
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')

class TextPostProcessingManager:
    
    '''
    takes in the label of the data and language code and pass it to the corresponding language/data preprocessor. The main class that the calling code will be interacting with.
    '''

    def __init__(self, label: str, language: str='') -> None:
       
        '''
        label: the data name to be checked and see if there are any special post-processing required to be done on the annotation
        language: language of the dataset
        '''

        self.label = label
        self.language = language


    def process_data(self, text: str) -> str:
        
        '''
        depending on the label and language, does the corresponding post-processing of the text 
        '''

        if self.language == 'eng':
            return TextPostProcessingEN().process(text=text)
        
        elif self.language == 'zh':
            # does not matter if the chinese is simplified or traditional
            return TextPostProcessingCJK().process(text=text)
        
        elif self.language == 'cmn':
            return TextPostProcessingZHSimplified().process(text=text)
            
        elif self.language == 'cmn_Hant':
            return TextPostProcessingZHTraditional().process(text=text)
        
        elif self.language == 'vie':
            return TextPostProcessingVI().process(text=text)
        
        elif self.language == 'tam':
            return TextPostProcessingTA().process(text=text)
        
        elif self.language == 'tgl':
            return TextPostProcessingTL().process(text=text)
        
        elif self.language == 'ind':
            return TextPostProcessingLatin().process(text=text)
        
        elif self.language == 'tha':
            return TextPostProcessingTH().process(text=text)
        
        else:
            # defaults to base, no processing, if text_preprocessing_language is ''
            return TextPostProcessingBase().process(text=text)


class TextPostProcessingBase:
    
    '''
    generic class to post-process dataset which has no preprocessing at all
    '''

    def __init__(self) -> None:
        pass


    def process(self, text: str) -> str:
        '''
        main method to normalise the latin annotation 
        '''

        return text.lstrip().rstrip()


class TextPostProcessingLatin:
    
    '''
    generic class to post-process dataset that uses the latin characters, may get inherited if there are specific preprocessing require by the specific dataset

    Latin alphabets: 26 characters used in english
    '''

    def __init__(self) -> None:
        pass


    def process(self, text: str) -> str:
        
        '''
        main method to normalise the latin annotation 
        '''

        # remove unicode characters
        text_en = text.encode("ascii", "ignore")
        text = text_en.decode()
        
        # keep only certain characters
        clean_text = re.sub(r'[^A-Za-z0-9#\' ]+', ' ', text)
        
        # replace hyphen with space because hyphen cannot be heard
        clean_text = clean_text.replace('-', ' ')
        
        # convert multiple spaces into only one space
        clean_text = ' '.join(clean_text.split())

        # returns the preprocessed text
        return clean_text.upper()
    

class TextPostProcessingEN(TextPostProcessingLatin):
    
    '''
    inherits the TextPostProcessingLatin class to do further post-processing for english language
    '''

    def __init__(self) -> None:
        super().__init__()


    def get_number_from_text(self, text: str) -> str:
        '''
        convert all the text form of the number into digit form to suit whisper decoding
        '''
        try:
            t2d = text2digits.Text2Digits()
            text = t2d.convert(text)
        except decimal.InvalidOperation:
            logging.warning("decimal.InvalidOperation ERROR, skipping transformation ...")
            pass

        return text

    
    def process(self, text: str) -> str:
        
        '''
        main method to normalise the english annotation, overrides the TextPostProcessingLatin implementation
        '''

        # remove unicode characters
        text_en = text.encode("ascii", "ignore")
        text = text_en.decode()
        
        # keep only certain characters
        clean_text = re.sub(r'[^A-Za-z0-9#\' ]+', ' ', text)

        # convert text form of number to digits
        clean_text = self.get_number_from_text(text=clean_text)
        
        # replace hyphen with space because hyphen cannot be heard
        clean_text = clean_text.replace('-', ' ')
        
        # convert multiple spaces into only one space
        clean_text = ' '.join(clean_text.split())

        # returns the preprocessed text
        return clean_text.upper()
    

class TextPostProcessingTL(TextPostProcessingLatin):
    
    '''
    inherits the TextPostProcessingLatin class to do further post-processing for tagalog language
    '''

    def __init__(self) -> None:
        super().__init__()

    
    def process(self, text: str) -> str:
        
        '''
        main method to normalise the tagalog annotation, overrides the TextPostProcessingLatin implementation
        '''
        
        # keep only certain characters, extends the n tilde in the tagalog language
        clean_text = re.sub(r'[^A-Za-z0-9#Ññ\' ]+', ' ', text)
        
        # replace hyphen with space because hyphen cannot be heard
        clean_text = clean_text.replace('-', ' ')
        
        # convert multiple spaces into only one space
        clean_text = ' '.join(clean_text.split())

        # returns the preprocessed text
        return clean_text.upper()
    

class TextPostProcessingCJK:
    
    '''
    generic class to post-process dataset that uses the chinese/japanese/korean/cantonese characters, may get inherited by other classes
    '''

    def __init__(self) -> None:

        self.cjk_ranges = [
            ( 0x0030,  0x0039), # numerals
            ( 0x4E00,  0x62FF),
            ( 0x6300,  0x77FF),
            ( 0x7800,  0x8CFF),
            ( 0x8D00,  0x9FCC),
            ( 0x3400,  0x4DB5),
            (0x20000, 0x215FF),
            (0x21600, 0x230FF),
            (0x23100, 0x245FF),
            (0x24600, 0x260FF),
            (0x26100, 0x275FF),
            (0x27600, 0x290FF),
            (0x29100, 0x2A6DF),
            (0x2A700, 0x2B734),
            (0x2B740, 0x2B81D),
            (0x2B820, 0x2CEAF),
            (0x2CEB0, 0x2EBEF),
            (0x2F800, 0x2FA1F)
        ]

    def is_cjk(self, char):
        
        char = ord(char)
        for bottom, top in self.cjk_ranges:
            if char >= bottom and char <= top:
                return True
        return False


    def process(self, text: str) -> str:
        
        '''
        main method to filter the cjk annotation
        '''

        return ' '.join(filter(self.is_cjk, text))


class TextPostProcessingZHTraditional(TextPostProcessingCJK):
    
    '''
    inherits the TextPostProcessingCJK class to do further post-processing for traditional chinese language
    '''

    def __init__(self) -> None:
        super().__init__()

    
    def process(self, text: str) -> str:
        '''
        main method to filter the cjk annotation
        '''

        return HanziConv.toTraditional(' '.join(filter(self.is_cjk, text)))
    

class TextPostProcessingZHSimplified(TextPostProcessingCJK):
    
    '''
    inherits the TextPostProcessingCJK class to do further post-processing for simplified chinese language
    '''

    def __init__(self) -> None:
        super().__init__()

    
    def process(self, text: str) -> str:
        '''
        main method to filter the cjk annotation
        '''

        return HanziConv.toSimplified(' '.join(filter(self.is_cjk, text)))
    

class TextPostProcessingTH:
    
    '''
    generic class to post-process dataset that uses the thai characters
    '''

    def __init__(self) -> None:

        self.th_ranges = [
            (0x0030, 0x0039), # numerals
            (0x0E01, 0x0E0F),
            (0x0E10, 0x0E1F),
            (0x0E20, 0x0E2F),
            (0x0E30, 0x0E3F),
            (0x0E40, 0x0E4F),
            (0x0E50, 0x0E5B)
        ]

    def is_th(self, char):
        char = ord(char)
        for bottom, top in self.th_ranges:
            if char >= bottom and char <= top:
                return True
        return False


    def process(self, text: str) -> str:
        '''
        main method to filter the thai annotation
        '''

        return ' '.join(filter(self.is_th, text))
    

class TextPostProcessingVI:
    
    '''
    generic class to post-process dataset that uses the vietnamese characters
    '''

    def __init__(self) -> None:

        self.vi_ranges = [
            (0x0020, 0x002F),
            (0x0030, 0x0039),
            (0x003A, 0x0040), 
            (0x0041, 0x005A),
            (0x005B, 0x0060),
            (0x0061, 0x007A),
            (0x007B, 0x007E),
            (0x00C0, 0x00C3),
            (0x00C8, 0x00CA),
            (0x00CC, 0x00CD),
            (0x00D0, 0x00D0),
            (0x00D2, 0x00D5),
            (0x00D9, 0x00DA),
            (0x00DD, 0x00DD),
            (0x00E0, 0x00E3),
            (0x00E8, 0x00EA),
            (0x00EC, 0x00ED),
            (0x00F2, 0x00F5),
            (0x00F9, 0x00FA),
            (0x00FD, 0x00FD),
            (0x0102, 0x0103),
            (0x0110, 0x0111),
            (0x0128, 0x0129),
            (0x0168, 0x0169),
            (0x01A0, 0x01B0),
            (0x1EA0, 0x1EF9)
        ]


    def remove_punct_en(self, sentence: str) -> str:
        sentence = sentence.replace('-', ' ')
        return sentence.translate(str.maketrans('', '', string.punctuation))
    

    def is_vi(self, char):
        char = ord(char)
        for bottom, top in self.vi_ranges:
            if char >= bottom and char <= top:
                return True
        return False


    def process(self, text: str) -> str:
        
        '''
        main method to filter the vietnamese annotation
        '''

        text = self.remove_punct_en(''.join(filter(self.is_vi, text.upper())))

        return text.lstrip().rstrip()
    

class TextPostProcessingTA:
    
    '''
    generic class to post-process dataset that uses the tamil characters
    '''

    def __init__(self) -> None:

        self.ta_ranges = [
            (0x0020, 0x0020), # spacing
            (0x0B82, 0x0B83),
            (0x0B85, 0x0B8A),
            (0x0B8E, 0x0B8F),
            (0x0B90, 0x0B90),
            (0x0B92, 0x0B95),
            (0x0B99, 0x0B9A),
            (0x0B9C, 0x0B9C),
            (0x0B9E, 0x0B9F),
            (0x0BA3, 0x0BA4),
            (0x0BA8, 0x0BAA),
            (0x0BAE, 0x0BAF),
            (0x0BB0, 0x0BB9),
            (0x0BBE, 0x0BBF),
            (0x0BC0, 0x0BC2),
            (0x0BC6, 0x0BC8),
            (0x0BCA, 0x0BCD),
            (0x0BD0, 0x0BD0),
            (0x0BD7, 0x0BD7),
            (0x0BE6, 0x0BEF),
            (0x0BF0, 0x0BFA)
        ]

    def is_ta(self, char):
       
        char = ord(char)
        for bottom, top in self.ta_ranges:
            if char >= bottom and char <= top:
                return True
        return False


    def process(self, text: str) -> str:
        
        '''
        main method to filter the tamil annotation
        '''

        text = ''.join(filter(self.is_ta, text))
        # text = ' '.join(filter(self.is_ta, text))

        return text.lstrip().rstrip()
        #return text.strip()


if __name__ == "__main__":

    pass
