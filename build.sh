 go build -o libserver.so -buildmode=c-shared server.go
  g++ -o tree_extractor c2.cpp `pkg-config --cflags --libs opencv4`
  gcc -o imghash imghash.c -L. -lserver
  LD_LIBRARY_PATH=/lib:/usr/lib/:/home/rafa/programacion/cand_doctorado/loop-detection/imgdiff2/imgfingerprint/   ./imghash
