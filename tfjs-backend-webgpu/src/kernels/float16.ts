/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {UnaryOpProgram} from './unary_op_webgpu';

const TO_FLOAT16 = `return float(float16(x));`;

export function float16(input: TensorInfo, backend: WebGPUBackend): TensorInfo {
  const program = new UnaryOpProgram(input.shape, TO_FLOAT16);
  const output = backend.makeOutputArray(input.shape, 'float16');
  return backend.compileAndRun(program, [input], output);
}
