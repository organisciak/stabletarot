import pkg_resources
areffont = pkg_resources.resource_filename(__name__, 'assets/aref.otf')
poppinsfont = pkg_resources.resource_filename(__name__, 'assets/Poppins-Regular.ttf')
contentfilter = pkg_resources.resource_filename(__name__, 'assets/expletives.json')

import json
import openai
import time
import re
from io import BytesIO
import random
import base64
import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
from IPython.display import HTML, display
import textwrap

'''
import diffusers
from ipywidgets import interactive
import ipywidgets as widgets

import requests

from diffusers import StableDiffusionPipeline
from huggingface_hub import notebook_login

'''

class TarotGenerator():

    def __init__(self, topic="Colorado", device="cuda", pipe=None, openai_key_path=None):
        self.topic = topic
        self.device = device
        self.cards = []

        if openai_key_path:
            openai.api_key_path = openai_key_path

        with open(contentfilter) as f:
            self.stopwords = json.load(f)['expletives']
        self.pipe = pipe # pass intialized diffusers pipe - if generating images

    def generate_cards(self, starters=[], model='davinci', temperature=0.75, include_starters=False,save_cards=True):
        prompt=f'''A Tarot Deck themed After {self.topic}, with 78 {self.topic}-themed cards.

## Original cards
1. The Magician
2. The High Priestess
3. The Empress
4. Ace of Swords
5. Knight of Wands
6. Queen of Pentacles
7. King of Cups
8. The Chariot
9. Strength
10. The Hermit
11. Wheel of Fortune
12. The Hierophant

## {self.topic}-themed parody of the original cards
1.'''
        if len(starters):
            starter_prompts = "\n".join([f'{i+1}. {x}' for i, x in enumerate(starters)]).strip().strip('1.')
            prompt+=starter_prompts

        results = openai.Completion.create(
        engine = model,
        temperature = temperature,
        presence_penalty = 1.2,
        prompt = prompt,
        max_tokens = 140
        )
        cards = []
        for y in results.choices[0].text.split('\n'):
            z = re.sub('^\d+\. *', '', y).strip()
            for sw in self.stopwords:
                if sw in z:
                    print('[Content filtered]')
                    continue
            if z.startswith('#'):
                continue
            if len(z) > 4:
                cards.append(z)
        if len(cards) < 4:
            print("format failed, trying again; 5s pause to kill")
            time.sleep(5)
            cards = self.generate_cards(starters, model, temperature, save_cards=False)

        # todo ask to confirm cards
        if save_cards:
            self.cards += cards
        return cards

    def _get_or_init_pipe(self):
        if not self.pipe:
            from diffusers import StableDiffusionPipeline
            self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", 
                                                                revision="fp16",
                                                                torch_dtype=torch.float16,
                                                                use_auth_token=True).to(self.device)
            # we don't want inappropriate images, but this checker only 
            # gives false positives with these prompts
            def dummy_checker(images, **kwargs): return images, False 
            self.pipe.safety_checker=dummy_checker
        return self.pipe

    def draw(self, n=1):
        ''' Draw a card from the deck'''
        if len(self.cards) == 0:
            raise Exception("Generate Cards first")
        cards = self.cards.copy()
        random.shuffle(cards)
        return cards[:n]

    def _add_title(self, card, image):
        ''' Add card name to image'''
        im1 = image.copy()
        canvas = ImageDraw.Draw(im1)

        lr_margins = 60
        bottom_margin = 20
        box_h = 80
        shape = [lr_margins, im1.height-box_h-bottom_margin, 
                 im1.width-lr_margins, im1.height-bottom_margin]
        canvas.rounded_rectangle(shape, fill ="white")

        fontsize = 32
        while True:
            myFont = ImageFont.truetype(areffont, fontsize)
            _,_,w, h = canvas.textbbox((0,0), card,font=myFont)
            if w > im1.width-lr_margins*2:
                fontsize -= 1
            else:
                break
        canvas.text(((im1.width-w)/2,im1.height-h/2-65), card, fill="black", font=myFont)
        return im1

    def _print_img(self, img, link_txt = None, return_html=False):
        ''' Base encode image and create a link for sending to a thermal printer via rawBT '''
        in_mem_file = BytesIO()
        img.save(in_mem_file, format = "jpeg")
        in_mem_file.seek(0)
        img_bytes = in_mem_file.read()
        enc_im = base64.b64encode(img_bytes).decode('ascii')

        url = f"rawbt:data:image/jpeg;base64,{enc_im}"
        if not link_txt:
            link_txt = f'<img src="data:image/jpeg;base64,{enc_im}" />'

        if return_html:
            return HTML(f"<a href='{url}' target='blank_'>{link_txt}</a>")
        else:
            return url

    def template(self, style):
        ''' if style is not present, it's assumed to be a template string '''
        if style == 'classic':
            template = "A exquisite black and white classic tarot card design for {}"
        elif style == 'modern':
            template = "A modern black and white tarot card design for {}"
        elif style == 'blackink':
            template = 'Modern tarot card illustration of "{}" by Derek Myers, in a fine-tip black ink illustration style, white background, fine detail, stippling, symmetrical'
        elif style=='crisp':
            template = 'Tarot card design for "{}" in a crisp and hip black and white flat line art vector style, aesthetic, captivating, clean and modern, fine detail, symmetrical'
        elif style == 'mucha':
            template = '"{}" tarot card featuring border illustrations, symmetrical, black and white, incredibly detailed art, Alphonse mucha painting'
        else:
            template = style
        return template

    def generate_card_img(self, card, style='mucha', steps=100,
                          scale=7.5, height=768, width=512, 
                          save_location='/content/drive/MyDrive/AI/Stable/tarot',
                          thermal_print=False):
        prompt = self.template(style).format(card)
        seed, image = self._gen_image(prompt, height=height, 
                                      width=width, steps = steps,
                                      return_seed=True)
        image = self._add_title(card, image)

        if save_location:
            outdir = Path(save_location)
            outdir.mkdir(exist_ok=True)
            clean_prompt = "".join([x for x in list(prompt) if x.isalpha()])
            fname = outdir / f"{clean_prompt}-{seed}.png"
            image.save(fname)
        if thermal_print:
            return self._print_img(image, card.upper(), return_html=True)
        else:
            return image

    def _gen_image(self, prompt, steps=100, scale=7.5, height=768, width=512, return_seed=True):
        seed = np.random.randint(10**4)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.autocast(self.device):
            image = self._get_or_init_pipe()(prompt, height=height, width=width, num_inference_steps = steps, scale=scale, generator=generator)["sample"][0]
        if return_seed:
            return seed, image
        else:
            return image

    def generate_reading_img(self, cards, interpretations, w=512, h=768, fontsize=18,
                             scale_font_up=True, thermal_print=False, scale_height=False):
        ''' Lay out interpretations onto an image.
        If scale_height is False, lay out the reading on one w x h card, adjusting fontsize
            and wrapping.
        If scale_height is True, use the default fontsize, width, and wrapping, and make the reading
        longer if necessary. Ignores scale_font_up and h arguments.

        scale_font_up=False keeps font from getting bigger - only smaller. '''
        if type(cards) == str:
            cards = [cards]
        if type(interpretations) == str:
            interpretations = [interpretations]
        wrapwidth = 50
        lr_margin = 20
        tb_margin = 60

        img = Image.new(mode='RGB', size=(w, h), color='white')
        draw = ImageDraw.Draw(img)

        full_msg = "\n\n".join([f"{card.upper()} represents{msg}" for card, msg in zip(cards, interpretations)])
        to_draw = "TAROT READING\n\n" + "".join(full_msg)
        paragraphs = to_draw.splitlines()

        maxadjusts = 15
        if scale_height == False:
            while True:
                font = ImageFont.truetype(font=poppinsfont, size=fontsize)
                to_draw_wrapped = "\n".join([textwrap.fill(p, wrapwidth) for p in paragraphs])
                _,_,textboxw, textboxh = draw.textbbox((0,0), text=to_draw_wrapped, font=font)
                # adjust word wrapping first, then adjust font if needed
                if maxadjusts <= 0:
                    break
                if (w-lr_margin*2-textboxw) > 60:
                    wrapwidth += 1
                    maxadjusts -= 1
                elif (textboxw+lr_margin*2) > w:
                    wrapwidth -= 1
                    maxadjusts -= 1
                elif textboxh > (h-tb_margin*2):
                    fontsize -= 1
                    maxadjusts -= 1
                elif (h-textboxh-tb_margin*2) > 60:
                    if scale_font_up:
                        fontsize += 1
                        maxadjusts -= 1
                    else:
                        break
                else:
                    break
        else:
            font = ImageFont.truetype(font=poppinsfont, size=fontsize)
            while True:
                to_draw_wrapped = "\n".join([textwrap.fill(p, wrapwidth) for p in paragraphs])
                _,_,textboxw, textboxh = draw.textbbox((0,0), text=to_draw_wrapped, font=font)

                if (w-lr_margin*2-textboxw) > 60:
                    wrapwidth += 1
                elif (textboxw+lr_margin*2) > w:
                    wrapwidth -= 1
                else:
                    break
            img = img.resize((w, textboxh+tb_margin*2))
            draw = ImageDraw.Draw(img)
            
        draw.text((w//2-textboxw//2, tb_margin),to_draw_wrapped,"black",font=font, align='left')
        if thermal_print:
            return self._print_img(img, "INTERPRETATION", return_html=True)
        else:
            return img

    def draw_and_gen(self, n=3, style='mucha', engine='curie', steps=40,
                     interpretations=True, thermal_print=False):
        cards = self.draw(n)

        ims = []
        if interpretations:
            interpretations = self.interpretations(cards, engine=engine)
            im = self.generate_reading_img(cards, interpretations, fontsize=24,
                                            scale_height=True, thermal_print=thermal_print)
            ims.append(im)
            display(im)

        for card in cards:
            im = self.generate_card_img(card,
                                        style=style,
                                        steps=steps,
                                        thermal_print=thermal_print)
            ims.append(im)
            display(im)
        return ims

    def interpretation(self, card, n=0, engine='curie', temperature=0.75):
        if n==0:
            lede = ""
        elif n == 1:
            lede = " This card represents where you've been."
        elif n == 2:
            lede = " This card represents where you are."
        elif n == 3:
            lede = " This card represents where you're going."
        else:
            lede = " This card is the wildcard."

        prompt = f'''TAROT CARD READING
        You drew this {self.topic}-themed card: {card}.{lede}

        Interpretation:

        The card {card} represents'''

        results = openai.Completion.create(engine = engine,
        temperature = temperature,
        prompt = prompt,
        presence_penalty=1,
        stop='\n\n',
        max_tokens = 150
        )

        interpretation = ".".join(results.choices[0].text.split('.')[:-1]) + '.'

        return interpretation

    def interpretations(self, cards, engine='curie', temperature=0.8):
        ''' Generate interpretations for multiple cards '''
        interpretations = []
        for i, card in enumerate(cards):
            interpretations += [self.interpretation(card, n=i+1, engine=engine,
                                                    temperature=temperature)]
        return interpretations