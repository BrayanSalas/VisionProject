#include <stdio.h>
#include <stdlib.h>

void prepare_dataset(char *dataset_name, char *preparing_type)
{
	char command[70];
	char* python_interpreter = "python ./dataset_processing.py";

	sprintf(command, "%s %s %s", python_interpreter, dataset_name, preparing_type);
	system(command);
}