文件夹转视频
ffmpeg -framerate 20 -i ai/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p merge.mp4
生成帧差异图
ffmpeg -i 1.mp4 -vf "tblend=all_mode=difference,format=gray,geq=lum='if(gt(lum(X,Y),32),255,0)',colorchannelmixer=aa=0" -r 20 mask_output/%05d.png
