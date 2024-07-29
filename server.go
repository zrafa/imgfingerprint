
package main

import (
	"C"
    "fmt"
    "image"
    "math/big"
    "os"
    "strings"

    "imgfingerprint/imgsim"
    _ "image/gif"
    _ "image/jpeg"
    _ "image/png"
)

func main () {
}

//export imghash
func imghash(imgpath_tmp *C.char) *C.char {
	var imgPath string
	imgPath = C.GoString(imgpath_tmp)
        imgPath = strings.TrimSpace(imgPath)

        imgfile, err := os.Open(imgPath)
        if err != nil {
            fmt.Println("Error:", err)
        }
        img, _, err := image.Decode(imgfile)
        imgfile.Close() // Mover el defer aquí no es adecuado ya que sigue acumulándose en cada ciclo
        if err != nil {
            fmt.Println("Error:", err)
        }

        ahash := imgsim.AverageHash(img)
        //ahash := imgsim.DifferenceHash(img)
        ahashInt := new(big.Int)
        ahashInt.SetString(ahash.String(), 2)

	//var result big.Int
	//result.Set(ahashInt)
	return C.CString(ahashInt.String())
}

