// ==UserScript==
// @name         BingDall
// @namespace    http://tampermonkey.net/
// @version      2024-01-04
// @description  try to take over the world!
// @author       you
// @match        https://www.bing.com/images/*
// @icon         https://www.google.com/s2/favicons?sz=64&domain=bing.com
// @grant        none
// ==/UserScript==


(async function () {
    'use strict';

    // 定义保存图片到本地的异步函数
    async function saveImageAsBlob(url, name) {
        try {
            const response = await fetch(url);  // 等待获取响应
            const blob = await response.blob();  // 等待转换为Blob

            // 创建一个指向Blob的URL
            const blobUrl = URL.createObjectURL(blob);

            // 创建一个a元素用于下载
            const link = document.createElement('a');
            link.href = blobUrl;
            link.download = name;  // 设置下载的文件名

            // 模拟点击链接以触发下载
            document.body.appendChild(link);
            link.click();

            // 清理: 移除链接元素，撤销Blob URL
            document.body.removeChild(link);
            URL.revokeObjectURL(blobUrl);

            console.log("Image should have been downloaded:", name);
        } catch (err) {
            console.error('下载图片时发生错误:', err);
        }
    }

    async function saveTextAsBlob(text, name) {
        // 创建Blob对象，指定内容和类型
        const blob = new Blob([text], { type: 'text/plain' });

        // 创建一个指向Blob的URL
        const blobUrl = URL.createObjectURL(blob);

        // 创建一个a元素用于下载
        const link = document.createElement('a');
        link.href = blobUrl;
        link.download = name; // 设置下载的文件名

        // 模拟点击链接以触发下载
        document.body.appendChild(link);
        link.click();

        // 清理：移除链接元素，撤销Blob URL
        document.body.removeChild(link);
        URL.revokeObjectURL(blobUrl);

        console.log("Text file should have been downloaded:", name);
    }

    async function checkAndOperate() {
        const element = document.querySelector('.gil_err_mt');

        if (element) {
            // 如果文本内容匹配，则打印错误信息
            console.error('错误：无法再提交任何提示');
            setTimeout(() => {
                window.location.href = 'https://www.bing.com/images/create';
            }, 1000 * 180);
            // 在发现错误后跳转到指定的URL
            return;
        }

        var targetElement = document.querySelector(".des_attr_dal_txt");
        // 获取input元素
        var inputElement = document.getElementById("sb_form_q");

        if (true) {
            var images = document.querySelectorAll("img.mimg, img.gir_mmimg");
            // 从localStorage获取已下载的图片URL集合
            var downloadedImages = new Set(JSON.parse(localStorage.getItem('downloadedImages') || '[]'));

            for (let i = 0; i < images.length; i++) {
                var img = images[i];
                var imgUrl = img.src || img.getAttribute('src');
                if (imgUrl) {
                    imgUrl = imgUrl.replace(/\?.*$/, '?pid=ImgGn');
                    var promptText = inputElement ? inputElement.value : '';
                    // 防止重复
                    if (!downloadedImages.has(imgUrl)) {
                        downloadedImages.add(imgUrl);

                        var urlParts = window.location.href.split('/');
                        var urlId = urlParts[urlParts.length-1];
                        await saveImageAsBlob(imgUrl, `${urlId}_${i}.jpg`);

                        await saveTextAsBlob(promptText, `${urlId}_${i}.txt`);
                        // 仅在循环的第一次迭代中保存当前URL
                        if (i === 0) {
                            await saveTextAsBlob(window.location.href, `${urlId}_DallURL.txt`);
                        }
                    }
                }
            }

            // 更新localStorage中的已下载图片集合
            localStorage.setItem('downloadedImages', JSON.stringify([...downloadedImages]));

            inputElement.value = "A high-quality, fullbody figurine,  It shows an anime beautiful girl with side ponytail in a fantastical aquarium greenhouse setting, lolita incorporate fractal crystal design, in sapphire and white tones.Wearing white stockings and boots, The artwork is infused with a psychedelic, dreamlike quality, resembling the work of a skilled anime illustrator, vibrant in color and rich in detail. avoiding the oversimplified chibi style.";//generateArtPrompt(inputElement);

            var clickButton = document.querySelector("#create_btn_c");
            if (clickButton) {
                clickButton.click();
                console.log("Button clicked!");
            } else {
                console.log("#create_btn_c not found");
            }
        }
        setTimeout(() => {
            window.location.reload();
        }, 1000 * 30);
    }

    function generateArtPrompt(inputElement) {
        // 定义颜色和场景选项
        const colors = [
            "blue", // 原有颜色
            //"red", // 原有颜色
            //"green", // 原有颜色
            //"lavender", // 薰衣草色
        ];
        const scenes = [

            "Sapphire crystal aquarium",
            "fractal_aquarium", // 分形水族馆
            "fractal_forest", // 森林
            "fractal_greenhouse", // 分形温室
            "fractal_world_tree", // 世界树
            "crystal_world", // 水晶世界
            "fractal_sky_islands", // 天空岛屿
            "fractal_starry_night", // 星夜
            "wisteria", // 紫藤
            "fractal_enchanted_forest", // 魔法森林
            "fractal_volcanic_realm", // 火山领域
            "fractal_desert_oasis", // 沙漠绿洲
            "fractal_moonlit_garden", // 月光花园
            "fractal_crystal_palace", // 水晶宫殿
            "fractal_cherry_blossom_forest", // 樱花森林
            "fractal_starlit_tower", // 星光塔
            "fractal_fairy_lake", // 仙女湖
            "fractal_island_of_dreams", // 梦之岛
            "fractal_misty_castle", // 雾中城堡
            "fractal_rainbow_bridge", // 彩虹桥
            "fractal_gateway_of_time", // 时间之门
            "fractal_ice_queen's_palace", // 冰雪女王的宫殿
            "fractal_magical_library", // 魔法图书馆
            "fractal_illusion_valley", // 幻影谷
            "fractal_secret_garden_gate", // 秘密花园之门
            "fractal_star_observatory", // 星星观测站
            "fractal_floating_crystal_islands", // 漂浮的水晶岛
            "fractal_mystical_fairy_cave", // 神秘的仙女洞
            "fractal_magic_treehouse", // 魔法树屋
            "fractal_amber_forest", // 琥珀森林
            "fractal_silver_mountain", // 银色山脉
            "crystal_fountain_garden", // 水晶喷泉花园
            "fractal_phantom_castle", // 幻影城堡
            "fractal_glowing_mushroom_valley", // 发光蘑菇谷
            "fractal_diamond_rainforest", // 钻石雨林
            "fractal_aurora_lake", // 极光湖
            "fractal_mirror_labyrinth", // 镜像迷宫
            "fractal_floating_gardens", // 漂浮花园
            "fractal_underwater_city", // 水下城市
            "fractal_spiral_towers", // 螺旋塔楼
            "fractal_lightning_temple", // 闪电寺庙
            "fractal_frost_flower_meadow", // 霜花草甸
            "fractal_sunset_vista", // 日落景观
            "fractal_coral_reef_palace", // 珊瑚礁宫殿
            "fractal_galactic_nebula", // 星系星云
            "fractal_fiery_chasm", // 火热裂谷
            "fractal_emerald_vineyard", // 翡翠葡萄园
            "fractal_silken_city", // 丝绸之城
            "fractal_phoenix_nest", // 凤凰巢
            "crystal_cavern_lagoon", // 水晶洞穴泻湖
            "fractal_petrified_forest", // 石化森林
            "fractal_luminous_canyon", // 发光峡谷
            "fractal_sapphire_meadows", // 蓝宝石草甸
            "fractal_pearlescent_coast", // 珍珠光泽海岸
            "fractal_eclipse_valley", // 日食谷
            "fractal_ruby_falls", // 红宝石瀑布
            "fractal_obsidian_cliffs", // 黑曜石崖
            "fractal_galactic_garden", // 银河花园
            "fractal_ivory_towers", // 象牙塔
            "fractal_mirage_oasis", // 海市蜃楼绿洲
            "fractal_twilight_grove", // 暮光林地
            "fractal_opal_harbor", // 蛋白石港
            "fractal_forgotten_temple", // 被遗忘的寺庙
            "fractal_zen_garden", // 禅意花园
            "fractal_cosmic_falls", // 宇宙瀑布
            "fractal_woven_silk_fields", // 编织丝绸田野
            "fractal_meteor_shower_lake", // 流星雨湖
            "fractal_ether_forest", // 以太森林
            "fractal_blooming_nightshade_grove", // 夜影花开林
            "luminous_orchid_valley", // 发光兰花谷
            "twilight_jasmine_archipelago", // 暮色茉莉群岛
            "everglow_lotus_ponds", // 永光莲花池
            "whispering_bamboo_forest", // 耳语竹林
            "singing_flower_fields", // 歌唱花田
            "crystal_berry_glades", // 水晶浆果林中空地
            "nebula_vine_orchards", // 星云藤果园
            "floating_fern_isles", // 漂浮蕨类岛屿
            "mirrorleaf_lakes", // 镜叶湖
            "phoenix_feather_palms", // 凤凰羽毛棕榈
            "moonbloom_ivies", // 月光蔓藤
            "stardust_willow_wharf", // 星尘柳木码头
            "sunflare_thistle_maze", // 日耀蓟迷宫
            "embershade_redwood_circuit", // 烬影红木圈
            "aurora_fern_gully", // 极光蕨谷
            "glimmering_pine_summit", // 闪烁松木峰
            "dreamweaver's_bramble_realm", // 织梦者的荆棘领域
            "soulblossom_ravine", // 灵魂花峡谷
            "echoing_mint_cascades" // 回声薄荷瀑布
        ];

        //var basePrompt = "A mesmerizing anime-style wallpaper illustration, A 2D fractal anime girl with [color] hair and blue highlights game cg, an ice mage with white stockings and boots, in [scene], adorned with fractal flowers and fractal feathers and crystal, embodying a magical and surreal aesthetic.";
        //var basePrompt = "A vibrant anime-style illustration with a focus on natural [color] hues, depicting a cute fractal cloths design, haired girl in a aquarium greenhouse-like fantasy setting. The art is reminiscent of professional anime illustrators, rich in color and detail, infused with psychedelic, dreamlike elements."; //我全都要
        var basePrompt = "A vibrant anime-style illustration with a focus on natural [color] hues, depicting a cute, haired girl in a aquarium greenhouse-like fantasy setting. The art is reminiscent of professional anime illustrators, rich in color and detail, infused with psychedelic, dreamlike elements."; //水族馆版
        //var basePrompt = "A vibrant anime-style illustration with a focus on natural [color] hues, depicting a cute fractal cloths design, haired girl in a greenhouse-like fantasy setting. The art is reminiscent of professional anime illustrators, rich in color and detail, infused with psychedelic, dreamlike elements."; //分形版
        //var basePrompt = "An anime-style girl wallpaper illustration, [scene]. Her outfit and hair are infused with fractal patterns, blending the intricacies of mathematical models with anime design. The environment around her pulsates with life, embodying a fusion of natural and digital realms through fractal art. This illustration captures the essence of beauty born from the fusion of fractal art and anime culture, showcasing the possibilities when these two worlds collide.";
        // 从数组中随机选择颜色和场景
        const randomColor = colors[Math.floor(Math.random() * colors.length)];
        const randomScene = scenes[Math.floor(Math.random() * scenes.length)];

        // 替换基础提示中的占位符
        const customPrompt = basePrompt.replace("[color]", randomColor).replace("[scene]", randomScene);

        return customPrompt;
    }

    // 首次启动函数
    console.log("The script is running. To stop it, clear the timeout.");
    setTimeout(checkAndOperate, 1000);
})();