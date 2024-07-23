
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



char * imghash(const char *);





int main() {

    char path[1035];
    char ruta[1024];
    char *ruta2 = "/tmp/output.png";

    const char * p = "/home/rafa/programacion/cand_doctorado/loop-detection/imgdiff/chicas/png/image0.jpg.chica.jpg.png";
    const char * parte1 = "/home/rafa/programacion/cand_doctorado/loop-detection/imgdiff/chicas/png/image";
    const char * parte2 = ".jpg.chica.jpg.png";

    while(1) {
        // Leer la ruta de la imagen desde stdin
	/*
        printf("Enter image path: ");
        fflush(0);
        if (fgets(path, sizeof(path)-1, stdin) == NULL) {
            printf("Failed to read image path\n");
            break;
        }
        path[strcspn(path, "\n")] = 0;  // Eliminar el salto de línea
        printf("pasamos 1: ");
        fflush(0);

	char *resultado = imghash(path);
	*/
	char *resultado = imghash(p);

	double hashes[100];
	int indices[100];

	int i;
	double d=0;
	for (i=1; i<=8; i++) {
		sprintf(ruta,"%s%i%s",parte1, i, parte2);
//		convertir(ruta);
		//resultado = imghash(ruta2);
		resultado = imghash(ruta);
		d = d+atof(resultado);
        	printf("1.%i :\t\t%lf\n", i, atof(resultado));
		hashes[i] = atof(resultado);
		indices[i] = 1;
	}
        printf("1 :\t\t%lf\n", d/8.0);

	d=0;
	for (i=9; i<=16; i++) {
		sprintf(ruta,"%s%i%s",parte1, i, parte2);
//		convertir(ruta);
		//resultado = imghash(ruta2);
		resultado = imghash(ruta);
		d = d+atof(resultado);
        	printf("2.%i :\t\t%lf\n", i, atof(resultado));
		hashes[i] = atof(resultado);
		indices[i] = 2;
	}
        printf("2 : \t\t%lf\n", d/8.0);

	d=0;
	for (i=17; i<=24; i++) {
		sprintf(ruta,"%s%i%s",parte1, i, parte2);
		//convertir(ruta);
		//resultado = imghash(ruta2);
		resultado = imghash(ruta);
		d = d+atof(resultado);
        	printf("3.%i : \t\t%lf\n", i, atof(resultado));
		hashes[i] = atof(resultado);
		indices[i] = 3;
	}
        printf("3 : \t\t%lf\n", d/8.0);

	sprintf(ruta,"%s25%s",parte1, parte2);
	//convertir(ruta2);
	resultado = imghash(ruta);
        printf("res: \t\t%lf\n", atof(resultado));

	double diferencia = 10000000000000000;
	double res = atof(resultado);
	double tmp;
	int indice = 0;
	for (i=1; i<=24; i++) {
		tmp = (hashes[i] - res);
		if (tmp < 0) tmp=tmp*(-1);
		if (tmp < diferencia) {
			diferencia = tmp;
			indice = i;
		}
	}
	if (indice != 0) {
        	printf("el arbol es el %i \n", indices[indice]);
        	printf("hash: %lf.  hash BD: %lf \n", res, hashes[indice]);
	}

        // Imprimir el resultado
        printf("Average Hash (entero): %s", resultado);
        fflush(0);
	exit(0);
    }

    return 0;
}

