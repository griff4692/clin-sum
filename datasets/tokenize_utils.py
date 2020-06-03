import os
import sys
from nltk.tokenize import ToktokTokenizer
from newspaper import fulltext


# function to clean data based on observed leftover social media text from html
def clean(line):
    # https://raw.githubusercontent.com/Alex-Fabbri/Multi-News/master/data/scripts/prep_data.py
    line = line.strip().replace("newline_char", " ")
    line = line.replace("( opens in new window )", "")
    line = line.replace('click prev or next to continue viewing', '')
    line = line.replace("click to email this to a friend", "")
    line = line.replace("lick to share on whatsapp", "")
    line = line.replace("click to share on facebook", "")
    line = line.replace("share on facebook", "")
    line = line.replace("click to share on twitter", "")
    line = line.replace("click to share on pinterest", "")
    line = line.replace("click to share on tumblr", "")
    line = line.replace("click to share on google+", "")
    line = line.replace("feel free to share these resources in your social "
                        "media networks , websites and other platforms", "")
    line = line.replace("share share tweet link", "")
    line = line.replace("e-mail article print share", "")
    line = line.replace("read or share this story :", "")
    line = line.replace("share the map view in e-mail by clicking the share "
                        "button and copying the link url .     embed the map "
                        "on your website or blog by getting a snippet of html "
                        "code from the share button .     if you wish to "
                        "provide feedback or comments on the map , or if "
                        "you are aware of map layers or other "
                        "datasets that you would like to see included on our maps , "
                        "please submit them for our evaluation using this this form .", "")
    line = line.replace("share this article share tweet post email", "")
    line = line.replace("skip in skip x embed x share close", "")
    line = line.replace("share tweet pin email", "")
    line = line.replace("share on twitter", "")
    line = line.replace("feel free to weigh-in yourself , via"
                        "the comments section . and while you ’ "
                        "re here , why don ’ t you sign up to "
                        "follow us on twitter us on twitter .", "")
    line = line.replace("follow us on facebook , twitter , instagram and youtube", "")
    line = line.replace("follow us on twitter", "")
    line = line.replace("follow us on facebook", "")
    line = line.replace("play facebook twitter google plus embed", "")
    line = line.replace("play facebook twitter embed", "")
    line = line.replace("enlarge icon pinterest icon close icon", "")
    line = line.replace("follow on twitter", "")
    line = line.replace('Article Continued Below', '')
    line = line.replace("autoplay autoplay copy this code to your website or blog", "")
    return line


def tokenize_news(str, tokenizer):
    text_newspaper = tokenizer.tokenize(fulltext(str))
    return clean(' '.join(text_newspaper))


# function to extract main text from Wayback links and tokenize the text
def clean_archive_data(folder):
    toktok = ToktokTokenizer()
    if not os.path.exists(f"{folder}-cleaned"):
        os.makedirs(f"{folder}-cleaned")
    for count, file in enumerate(os.listdir(f"{folder}")):
        if count % 1000 == 0:
            print(count)
        file_data = open(f"{folder}/{file}", "r").read()
        try:
            text_newspaper = toktok.tokenize(fulltext(file_data))
            text_newspaper_cleaned = clean(" ".join(text_newspaper))
            with open(f"{folder}-cleaned/{file}", "w") as output:
                output.write(text_newspaper_cleaned)
        except:  # pylint: disable=W0702
            print(f"error with {file}", file=sys.stderr)


def clean_summary_str(s):
    s = s.lower()
    s = s.replace('<unk>', '')
    s = s.replace('`', '')
    s = s.replace('.', '')
    s = s.replace(',', '')
    s = s.replace(';', '')
    s = s.replace('\'', '')
    s = s.replace('\"', '')
    s = s.replace('(', '')
    s = s.replace(')', '')
    s = s.replace('-', ' ')
    s = s.replace('<p>', '')
    s = s.replace('</p>', '')
    s = s.replace('<t>', '')
    s = s.replace('</t>', '')
    s = s.replace('[!@#$]', '')
    return s
