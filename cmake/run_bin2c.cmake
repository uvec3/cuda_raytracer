execute_process(
    COMMAND "${BIN2C}" -c --type char --name embedded_optix_ir "${INPUT}"
    OUTPUT_FILE "${OUTPUT}"
    RESULT_VARIABLE res
)
if(NOT res EQUAL 0)
    message(FATAL_ERROR "bin2c failed with exit code ${res}")
endif()
