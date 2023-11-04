for %%a in (*.mkv *.mp4 *.avi) do (
    mkdir "%%~na"
    ffmpeg -i "%%a" -vf "select='gt(scene,0.3)',showinfo" -vsync 0 -q:v 2 -f image2 "%%~na/%%~na%%05d.jpg"
)
