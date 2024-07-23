
package main

import (
    "flag"
    "fmt"
    "image/png"
    "os"
    "math/big"

    "github.com/Nr90/imgsim"
)

func main() {
    // Definir el parámetro de la línea de comandos
    imgPath := flag.String("img", "", "Ruta a la imagen")
    flag.Parse()

    if *imgPath == "" {
        fmt.Println("Se requiere una ruta a la imagen")
        os.Exit(1)
    }

    imgfile, err := os.Open(*imgPath)
    if err != nil {
        panic(err)
    }
    defer imgfile.Close()

    img, err := png.Decode(imgfile)
    if err != nil {
        panic(err)
    }

    //    ahash := imgsim.AverageHash(img)
    //fmt.Println("Average Hash:", ahash)

    //dhash := imgsim.DifferenceHash(img)
    //fmt.Println("Difference Hash:", dhash)


    ahash := imgsim.AverageHash(img)
    ahashInt := new(big.Int)
    ahashInt.SetString(ahash.String(), 2)

    // fmt.Println("Average Hash (binario):", ahash)
    fmt.Println("\n\t\t\tAverage Hash (entero):", ahashInt)

    dhash := imgsim.DifferenceHash(img)
    dhashInt := new(big.Int)
    dhashInt.SetString(dhash.String(), 2)

    // fmt.Println("Difference Hash (binario):", dhash)
    fmt.Println("Difference Hash (entero):", dhashInt)
}

