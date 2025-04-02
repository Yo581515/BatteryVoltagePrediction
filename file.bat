@ECHO ON
@if "%URL%" equ "" , echo NO URL & set /p URL=[Enter video URL]
yt-dlp -v -c -o YTDLP/%%(channel)s/[%%(upload_date)s]%%(fulltitle).50s(%%(id)s)/[%%(upload_date)s]%%(fulltitle)s(%%(id)s) --add-metadata --concurrent-fragments 20 --cookies cookies-youtube-com.txt --merge-output-format mp4 --embed-thumbnail --embed-metadata --write-info-json --clean-infojson --write-comments --write-subs --sub-lang all --sub-format srt --write-description --write-thumbnail %URL%
@cmd /k