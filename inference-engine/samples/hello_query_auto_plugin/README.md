# Hello Query Auto Plugin C++ Sample {#openvino_inference_engine_samples_hello_query_auto_plugin_README}

This topic demonstrates how to run the Hello Query Auto Plugin sample application, which queries Inference Engine Auto Plugin and prints the supported metrics and default configuration values.
> **NOTE:** This topic describes usage of C++ implementation of the Query Auto Plugin Sample. 


To see quired information, run the following:
```sh
./hello_query_auto_plugin
```

## Sample Output

The application prints auto plugin with the supported metrics and default values for configuration parameters:

```
Device: AUTO
Metrics: 
        AVAILABLE_DEVICES : [  ]
        SUPPORTED_METRICS : [ AVAILABLE_DEVICES SUPPORTED_METRICS FULL_DEVICE_NAME SUPPORTED_CONFIG_KEYS OPTIMIZATION_CAPABILITIES ]
        FULL_DEVICE_NAME : AUTO
        SUPPORTED_CONFIG_KEYS : [ AUTO_DEVICE_PRIORITIES ]
        OPTIMIZATION_CAPABILITIES : [ CPU: FP32 FP16 INT8 BIN  ]
Default values for device configuration keys: 
        AUTO_DEVICE_PRIORITIES : ""
```
