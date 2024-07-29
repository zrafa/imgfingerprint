
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



char * imghash(const char *);


convertir (char * origen)
{
	char comando[256];
	//sprintf(comando, "convert %s -gravity North -region 100%cx50%c -fill black -colorize 100\% /tmp/output.png", origen, '%', '%');
	sprintf(comando, "convert %s -gravity Center -crop 20%cx100%c+0+0 +repage /tmp/output.png", origen, '%', '%');

	printf("%s \n", comando);
	fflush(0);
	system(comando);
}



int main() {

    char path[1035];
    char ruta[1024];
    char *ruta2 = "/tmp/output.png";

    const char * p = "/home/rafa/programacion/cand_doctorado/loop-detection/imgdiff/chicas/png/image0.jpg.chica.jpg.png";
    const char * parte1 = "chicas/i";
    const char * parte2 = ".jpg.png";

    while(1) {
	char *resultado;

	double hashes[100];
	int indices[100];

	int i;
	for (i=0; i<=102; i++) {
		sprintf(ruta,"%s%i%s",parte1, i, parte2);
		//convertir(ruta);
		//sprintf(ruta,"/tmp/output.png");
		resultado = imghash(ruta);
        	printf("1.%i :\t\t%lf\n", i, atof(resultado));
		hashes[i] = atof(resultado);
		indices[i] = i;
	}

	double diferencia;
	double res;
	double tmp;
	int indice;
	int j;

	for (j=85; j<=102; j++) {
		sprintf(ruta,"%s%i%s",parte1, j, parte2);
		//convertir(ruta);
		//sprintf(ruta,"/tmp/output.png");
		resultado = imghash(ruta);
		//printf("res: \t\t%lf\n", atof(resultado));

		diferencia = 10000000000000000;
		res = atof(resultado);
		indice = -1;
	for (i=0; i<=84; i++) {
		tmp = (hashes[i] - res);
		if (tmp < 0) tmp=tmp*(-1);
		if (tmp == 0)
			continue;
		if (tmp < diferencia) {
			diferencia = tmp;
			indice = i;
		}
	}
		if (indice != -1) {
        		printf("el arbol es el %i %i %i \n", j, indice, indices[indice]);
        		printf("hash: %lf.  hash BD: %lf \n", res, hashes[indice]);
		}
	}

        // Imprimir el resultado
        printf("Average Hash (entero): %s", resultado);
        fflush(0);
	exit(0);
    }

    return 0;
}

