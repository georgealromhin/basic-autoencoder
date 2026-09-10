# basic-autoencoder

A minimal example of an `AutoEncoder` layer configuration for [Deeplearning4j](https://deeplearning4j.konduit.ai/) (DL4J), a JVM-based deep learning library. This class defines a **denoising autoencoder**: a layer that learns to reconstruct its input after the input has been intentionally corrupted with noise, which forces the network to learn robust, meaningful features instead of just memorizing the identity function.

Note that this is a *layer configuration* class — it describes how the layer should be built and how much memory it needs, but the actual forward/backward pass math lives in DL4J's internal `org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder` class, which this configuration instantiates.

## How the code is organized

### Class declaration and Lombok annotations

```java
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class AutoEncoder extends BasePretrainNetwork {
```

The class uses [Lombok](https://projectlombok.org/) to avoid writing boilerplate by hand:
- `@Data` generates getters/setters, `equals()`, `hashCode()`, and `toString()`.
- `@NoArgsConstructor` generates an empty constructor (needed for deserialization).
- `@ToString(callSuper = true)` / `@EqualsAndHashCode(callSuper = true)` make sure fields from the parent class (`BasePretrainNetwork`) are included, not just this class's own fields.

It extends `BasePretrainNetwork`, which marks it as a layer type meant to be trained in an unsupervised, "pretraining" style — the hallmark of a classic autoencoder.

### Fields

```java
protected double corruptionLevel;
protected double sparsity;
```

- **`corruptionLevel`** — how much noise to inject into the input before reconstruction, from `0.0` (no corruption) to `1.0` (input fully corrupted). This is what makes it a *denoising* autoencoder: the network must reconstruct the clean input from a noisy version of it.
- **`sparsity`** — a sparsity penalty parameter that encourages the hidden layer's activations to stay small/sparse, which can improve the quality of learned features.

### Private constructor

```java
private AutoEncoder(Builder builder) {
    super(builder);
    this.corruptionLevel = builder.corruptionLevel;
    this.sparsity = builder.sparsity;
    initializeConstraints(builder);
}
```

The constructor is private and only called from the nested `Builder` class, following the [Builder pattern](https://en.wikipedia.org/wiki/Builder_pattern) that's standard across DL4J's layer configuration API. This keeps object construction consistent and validated.

### `instantiate(...)`

```java
@Override
public Layer instantiate(NeuralNetConfiguration conf, ...) {
    org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder ret =
        new org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder(conf, networkDataType);
    ...
    return ret;
}
```

This is where the configuration turns into a real, runnable layer. It creates the actual `AutoEncoder` layer implementation, wires up training listeners, sets its position (`layerIndex`) in the network, attaches the parameter array (weights/biases), and initializes those parameters via the layer's `ParamInitializer`.

### `initializer()`

```java
@Override
public ParamInitializer initializer() {
    return PretrainParamInitializer.getInstance();
}
```

Returns the strategy DL4J uses to lay out and initialize this layer's parameters (weights and biases) in memory. `PretrainParamInitializer` is the initializer shared by unsupervised pretraining layers.

### `getMemoryReport(...)`

```java
@Override
public LayerMemoryReport getMemoryReport(InputType inputType) {
    ...
}
```

Estimates how much memory this layer will need — for its parameters, updater state (e.g. momentum/Adam state), and working memory during training — given a particular input shape. This lets DL4J validate that a network configuration will fit in available memory *before* training starts, rather than failing partway through. The comments in the method explain the assumptions made (e.g. assuming the more memory-hungry unsupervised training mode, and duplicating the input when dropout is used).

### Nested `Builder` class

```java
public static class Builder extends BasePretrainNetwork.Builder<Builder> {
    private double corruptionLevel = 3e-1f;
    private double sparsity = 0f;
    ...
    public AutoEncoder build() {
        return new AutoEncoder(this);
    }
}
```

The fluent builder used to configure and create an `AutoEncoder` layer, e.g.:

```java
new AutoEncoder.Builder()
    .corruptionLevel(0.3)
    .sparsity(0.0)
    .nIn(784)
    .nOut(250)
    .build();
```

It defaults `corruptionLevel` to `0.3` and `sparsity` to `0.0`, and exposes `corruptionLevel(...)` / `sparsity(...)` methods that set the corresponding field and return `this` so calls can be chained. `build()` finally produces the immutable `AutoEncoder` layer configuration.

## Full source

```java

package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Autoencoder layer. Adds noise to input and learn a reconstruction function.
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class AutoEncoder extends BasePretrainNetwork {

    protected double corruptionLevel;
    protected double sparsity;

    // Builder
    private AutoEncoder(Builder builder) {
        super(builder);
        this.corruptionLevel = builder.corruptionLevel;
        this.sparsity = builder.sparsity;
        initializeConstraints(builder);
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
        org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder ret =
                        new org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder(conf, networkDataType);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return PretrainParamInitializer.getInstance();
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        //Because of supervised + unsupervised modes: we'll assume unsupervised, which has the larger memory requirements
        InputType outputType = getOutputType(-1, inputType);

        val actElementsPerEx = outputType.arrayElementsPerExample() + inputType.arrayElementsPerExample();
        val numParams = initializer().numParams(this);
        val updaterStateSize = (int) getIUpdater().stateSize(numParams);

        int trainSizePerEx = 0;
        if (getIDropout() != null) {
            if (false) {
                //TODO drop connect
                //Dup the weights... note that this does NOT depend on the minibatch size...
            } else {
                //Assume we dup the input
                trainSizePerEx += inputType.arrayElementsPerExample();
            }
        }

        //Also, during backprop: we do a preOut call -> gives us activations size equal to the output size
        // which is modified in-place by loss function
        trainSizePerEx += actElementsPerEx;

        return new LayerMemoryReport.Builder(layerName, AutoEncoder.class, inputType, outputType)
                        .standardMemory(numParams, updaterStateSize).workingMemory(0, 0, 0, trainSizePerEx)
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                        .build();
    }

    @AllArgsConstructor
    @Getter
    @Setter
    public static class Builder extends BasePretrainNetwork.Builder<Builder> {

        /**
         * Level of corruption - 0.0 (none) to 1.0 (all values corrupted)
         *
         */
        private double corruptionLevel = 3e-1f;

        /**
         * Autoencoder sparity parameter
         *
         */
        private double sparsity = 0f;

        public Builder() {}

        /**
         * Builder - sets the level of corruption - 0.0 (none) to 1.0 (all values corrupted)
         *
         * @param corruptionLevel Corruption level (0 to 1)
         */
        public Builder(double corruptionLevel) {
            this.setCorruptionLevel(corruptionLevel);
        }

        /**
         * Level of corruption - 0.0 (none) to 1.0 (all values corrupted)
         *
         * @param corruptionLevel Corruption level (0 to 1)
         */
        public Builder corruptionLevel(double corruptionLevel) {
            this.setCorruptionLevel(corruptionLevel);
            return this;
        }

        /**
         * Autoencoder sparity parameter
         *
         * @param sparsity Sparsity
         */
        public Builder sparsity(double sparsity) {
            this.setSparsity(sparsity);
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public AutoEncoder build() {
            return new AutoEncoder(this);
        }
    }
}
```
