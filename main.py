from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.core.text import LabelBase
from plyer import filechooser
import numpy as np
import joblib
from spectrum_utils import extract_spectrum_features

# 注册中文字体（SimHei.ttf）
LabelBase.register(name="SimHei", fn_regular="fonts/SimHei.ttf")


class SugarPredictor(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        self.img = Image(size_hint=(1, .6))
        self.add_widget(self.img)

        self.lbl = Label(text="请选择光谱图", font_name="SimHei", font_size=20, size_hint=(1, .1))
        self.add_widget(self.lbl)

        btn = Button(text="打开相册", font_name="SimHei", font_size=18, size_hint=(1, .15))
        btn.bind(on_press=lambda x: filechooser.open_file(on_selection=self.selected))
        self.add_widget(btn)

        self.model = joblib.load("model/spectrum_brix_model.pkl")

    def selected(self, selection):
        if not selection:
            return
        path = selection[0]
        self.img.source = path
        self.lbl.text = "分析中..."
        try:
            feat = extract_spectrum_features(path).reshape(1, -1)
            brix = float(self.model.predict(feat)[0])
            self.lbl.text = f"预测糖度：{brix:.2f} °Brix"
        except Exception as e:
            self.lbl.text = f"出错啦：{e}"


class BrixApp(App):
    def build(self):
        return SugarPredictor()


if __name__ == '__main__':
    BrixApp().run()
