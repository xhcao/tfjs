/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import {util} from '@tensorflow/tfjs-core';

import {getCoordsDataType} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export const RELU = 'return max(a, float16_t(0.0));';
export const RELU6 = 'return (a < float16_t(0.0)) ? float16_t(0.0) : min(float16_t(6.0), a);';
export const LINEAR = `return a;`;
export const ELU = `return (a >= float16_t(0.0)) ? a : (exp(a) - float16_t(1.0));`;

export const SIGMOID = `return float16_t(1.0) / (float16_t(1.0) + exp(float16_t(-1.0) * a));`;
export const ABS = `return abs(a);`;
export const SQUARE = `return a * a;`;
export const NEG = `return -a;`;
export const TANH = `
  float e2x = exp(float16_t(-2.0) * abs(a));
  return sign(a) * (float16_t(1.0) - e2x) / (float16_t(1.0) + e2x);
`;
export const EXP = `return exp(a);`;
export const LOG = `if (a < float16_t(0.0)) return float16_t(1.0)/float16_t(0.0);
  return log(a);`;

export class UnaryOpProgram implements WebGPUProgram {
  outputShape: number[];
  userCode: string;
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A'];
  workPerThread: number;
  workGroupSize: [number, number, number];

  constructor(outputShape: number[], op: string) {
    // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
    const workGroupSizeX = 128;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = outputShape;
    const size = util.sizeFromShape(this.outputShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    const fit = size % workGroupSizeX === 0;
    this.workPerThread = fit ? 1 : 2;
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    if (fit) {
      this.userCode = `
      float16_t unaryOperation(float16_t a) {
        ${op}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);
        float16_t a = A[index];
        setOutput(index, unaryOperation(a));;
      }
      `;
      this.shaderKey = `unary2${op}`;
    } else {
      const type = getCoordsDataType(this.outputShape.length);
      this.userCode = `
      float16_t unaryOperation(float16_t a) {
        ${op}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);

        for(int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;

          if(flatIndex < ${size}) {
            ${type} coords = getCoordsFromFlatIndex(flatIndex);

            float16_t a = getAAtOutCoords(coords);
            setOutput(flatIndex, unaryOperation(a));
          }
        }
      }
      `;
      this.shaderKey = `unary${op}${type}${size}`;
    }
  }
}
