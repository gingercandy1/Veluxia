### 项目使用方法

uv pip install -r src/core/speech/ACE_Step/requirements.txt


uv pip install -e "."


### 资产生成流程

角色设计
1. 设计角色的关键词
2. 批量生成图片（角色、场景）
3. 图片提高分辨率
4. 将图片转换为视频 
5. 推荐实现自动化生成关键词，生成一些不重要的角色

场景设计
1. 设计场景关键词
2. 批量生成图片
3. 导入到unity参考设计场景

角色对话面部
1. 编写对话内容
2. 根据对话内容角色生成声音
3. 根据对话内容生成角色面部表情

背景音乐
1. 生成背景音乐
2. 还有各种技能特效的声音


### Unity流程
1. 角色导入到Unity中
2. 角色和场景交互
3. 界面实现添加
4. 

### 第三方库
1. Qwen2.5-1.5B-Instruct (Apache协议)  # 优化提示词
2. FLUX.2-klein-4b-nvfp4 （Apache协议）   # 生成图片
3. Meta-Llama-3.1-8B （ Llama 3.1 社区许可协议） # 文本生成
4. LTX Video（Apache协议）  # 图片生成动画
5. Wan 2.2 （Apache协议）   # 图片生成动画
6. ACE-Step/Ace-Step1.5     # 生成音乐
7. Qwen3-TTS-12Hz-1.7B-VoiceDesign  (Apache协议) # 生成人物音频 


接下来
程序界面
1. 完善界面需要继续补充界面的功能，比如按钮

文本、对话
1. 需要生成故事剧情以及人物的对话流程，可以制作一个工具
2. 生成角色身份
3. 

声音、音乐
2. 添加人声阅读 https://github.com/QwenLM/Qwen3-TTS 【有待测试】
3. 添加音乐生成支持 https://github.com/ace-step/ACE-Step-1.5 【有待测试】


视频图像
2. 生成的视频需要解决循环问题。
3. 我希望生成一段图片中人物跟着阅读的视频，主要是为了让人物和声音口型能对上，我给一个声音和一张图片不知道是否能实现？


目标：
1. 先将主要角色概念图生成
2. 生成角色动画效果，比如一些常见的动作，比如跑、随意站立等效果
3. 生成主要角色战斗效果
4. 生成角色后然后生成一些简单的场景，让角色再场景中能跑起来。


prompt = (
    f"{prompt}, "
    "high quality, smooth motion, consistent character, "
    "anime style, 2D animation"
)





### 正派角色提示词
A distinctive cartoon character, full body standing pose, front view or three-quarter view, 
sharp and bold line art with thick heavy outlines, crisp ink-like strokes, dark sophisticated color palette with muted grays, 
deep blacks, cool tones and high-end grayish atmosphere, premium dark aesthetic, stylish and memorable design, 
clean white background, high contrast,  professional game character sheet, positive and likable appearance

A brave young girl, around 13-16 years old, determined and slightly melancholic expression, short messy dark hair with a small braid, 
big expressive eyes, wearing oversized patched cloak, simple linen dress with leather straps, worn boots, carrying a small dagger on her belt, 
full body standing pose, distinctive cartoon style, thick heavy black outlines, sharp crisp linework, dark sophisticated muted gray and cool tone color palette, 
premium dark aesthetic, clean white background, high contrast, professional character sheet


A burly middle-aged male blacksmith, around 38-45 years old, tall and muscular build, rough square face with thick beard and short messy brown hair, 
kind but tired eyes, wearing a dirty leather apron over bare chest with burn marks, thick leather bracers, heavy boots, holding a large hammer in one hand, 
full body standing pose, distinctive cartoon style, thick heavy black outlines, sharp crisp linework, dark sophisticated muted gray and cool tone color palette, 
premium dark aesthetic, clean white background, high contrast, professional character sheet

A sarcastic young adult female mage, around 26-30 years old, slim and elegant build, sharp intelligent eyes with a mocking expression, 
long straight black hair with silver streaks, pale skin, wearing a dark hooded mage robe with subtle glowing runes, leather belt with potion vials, 
high boots, holding a wooden staff, full body standing pose, distinctive cartoon style, thick heavy black outlines, sharp crisp linework, 
dark sophisticated muted gray and cool tone color palette, premium dark aesthetic, clean white background, high contrast, professional character sheet


### 反派提示词
Grotesque horror monster, full body standing pose, distinctive dark cartoon style, thick heavy black outlines, sharp crisp edgy linework, 
dark sophisticated muted gray palette with sickly accents, creepy and horrif    ying atmosphere, clean white background, high contrast, professional game enemy character sheet


A horrifying plant-human hybrid monster, human torso fused with twisted black vines and thorny flowers blooming from body, mushroom-like growths on head, 
long vine tentacles for arms, glowing sickly yellow eyes, tattered clothes, full body standing pose, distinctive dark cartoon style, thick heavy black outlines, 
sharp crisp linework, dark sophisticated muted gray and dark green tones, grotesque atmosphere, clean white background, high contrast, professional game enemy character sheet



### 动作提示词

walking forward naturally, smooth gait, arms swinging
running at moderate speed, dynamic motion
standing idle, subtle breathing motion, slight body sway
sitting down slowly, natural movement
standing up from chair, realistic motion

jumping in place, arms raised in joy
spinning around once, smooth rotation
kicking forward, martial arts style
punching forward, action pose
dancing rhythmically, hip hop style
throwing an object forward, full body follow-through


bowing respectfully, slow and deliberate motion
praying with hands clasped together
meditating, cross-legged, eyes closed, calm breathing
playing piano, fingers moving gracefully
writing with pen on paper, focused expression
reading a book, page turning slowly

talking on phone, expressive hand gestures
typing on keyboard quickly, focused expression
drinking from a cup, slow sip
eating with chopsticks, natural motion
looking at watch, checking time
opening a door and walking through


high quality, smooth motion, 
anatomically correct, stable body, consistent limbs, 
no morphing, no flickering, 

# 质量类
high quality, smooth animation, fluid motion, 
realistic movement, cinematic style

# 视角类
front view          # 正面
side view           # 侧面
close-up            # 特写
full body shot      # 全身

# 风格类
anime style         # 动漫风
3D rendered         # 3D渲染
cartoon style       # 卡通风
realistic           # 写实风



# 素材

蘑菇
A diverse cluster of bioluminescent mushrooms, various sizes and heights, 
irregular organic shapes, small glowing mushrooms cluster, hand-painted fantasy illustration, 
bioluminescent blue-white glow, detailed cap texture, visible brush strokes, painterly style, 
dark base with luminous tips, soft light emission, natural asymmetrical composition, 
some tilted and some upright, mystical forest floor element, white background, isolated, high detail.


卢苇草
reed grass with long slender leaves, gently swaying,
hand-painted illustration style, detailed brush strokes,
oil painting texture, visible paint texture on leaves,

deep blue and teal color palette, 
dark forest atmosphere, mystical glowing ambiance,
soft bioluminescent light catching the edges of leaves,
subtle rim lighting, blue-purple light from background,

delicate leaf details, fine vein texture on each blade,
layered depth, foreground element,
slightly transparent leaf tips,
organic natural shapes, asymmetric growth pattern,

white background, isolated element, clean edges,
game concept art style, cinematic fantasy,
high detail, high quality

树干/树枝
ancient twisted tree trunk with bare branches,
hand-painted illustration, detailed bark texture,
visible brush strokes, oil painting style,
deep blue purple atmospheric background glow,
dark silhouette with subtle blue-grey highlights,
mystical forest, cinematic fantasy game art,
white background, isolated, high detail

发光蘑菇
small glowing mushrooms cluster, 
hand-painted fantasy illustration,
bioluminescent blue-white glow, detailed cap texture,
visible brush strokes, painterly style,
dark base with luminous tips, soft light emission,
mystical forest floor element,
white background, isolated, high detail

藤蔓
hanging vine with small leaves, 
hand-painted illustration, detailed brush texture,
deep teal green tones, subtle blue rim lighting,
organic flowing shape, naturalistic curl,
painterly fantasy art style,
white background, isolated, high detail




SIZE_GROUP_LABELS = {
    "tiny":           "Tiny  ( < 1B )",
    "small":          "Small  ( 1B – 3B )",
    "medium":         "Medium  ( 4B – 9B )",
    "large":          "Large  ( 10B – 30B )",
    "xlarge":         "XLarge  ( 30B+ )",
    "custom":         "Custom Voice",
    "clone":          "Voice Clone",
    "design":         "Voice Design",
    "fast":           "Fast",
    "image-to-video": "Image to Video",
}