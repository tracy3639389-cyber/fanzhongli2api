import memoryManager, { registerMemoryPoolCleanup } from '../utils/memoryManager.js';
import { generateToolCallId } from '../utils/idGenerator.js';
import { setReasoningSignature, setToolSignature } from '../utils/thoughtSignatureCache.js';
import { getOriginalToolName } from '../utils/toolNameCache.js';

// 预编译的常量
const DATA_PREFIX = 'data: ';
const DATA_PREFIX_LEN = DATA_PREFIX.length;

// 高效的行分割器（LineBuffer）
class LineBuffer {
  constructor() {
    this.buffer = '';
    this.lines = [];
  }
  
  append(chunk) {
    this.buffer += chunk;
    this.lines.length = 0;
    
    let start = 0;
    let end;
    while ((end = this.buffer.indexOf('\n', start)) !== -1) {
      this.lines.push(this.buffer.slice(start, end));
      start = end + 1;
    }
    
    this.buffer = start < this.buffer.length ? this.buffer.slice(start) : '';
    return this.lines;
  }
  
  clear() {
    this.buffer = '';
    this.lines.length = 0;
  }
}

// 对象池逻辑
const lineBufferPool = [];
const getLineBuffer = () => {
  const buffer = lineBufferPool.pop();
  if (buffer) {
    buffer.clear();
    return buffer;
  }
  return new LineBuffer();
};
const releaseLineBuffer = (buffer) => {
  const maxSize = memoryManager.getPoolSizes().lineBuffer;
  if (lineBufferPool.length < maxSize) {
    buffer.clear();
    lineBufferPool.push(buffer);
  }
};

const toolCallPool = [];
const getToolCallObject = () => toolCallPool.pop() || { id: '', type: 'function', function: { name: '', arguments: '' } };
const releaseToolCallObject = (obj) => {
  const maxSize = memoryManager.getPoolSizes().toolCall;
  if (toolCallPool.length < maxSize) toolCallPool.push(obj);
};

function registerStreamMemoryCleanup() {
  registerMemoryPoolCleanup(toolCallPool, () => memoryManager.getPoolSizes().toolCall);
  registerMemoryPoolCleanup(lineBufferPool, () => memoryManager.getPoolSizes().lineBuffer);
}

function convertToToolCall(functionCall, sessionId, model) {
  const toolCall = getToolCallObject();
  toolCall.id = functionCall.id || generateToolCallId();
  let name = functionCall.name;
  if (sessionId && model) {
    const original = getOriginalToolName(sessionId, model, functionCall.name);
    if (original) name = original;
  }
  toolCall.function.name = name;
  toolCall.function.arguments = JSON.stringify(functionCall.args);
  return toolCall;
}

/**
 * 核心修改区域：解析并双重发送思考内容
 * 实现了“思维链分离与再处理”逻辑
 */
function parseAndEmitStreamChunk(line, state, callback) {
  if (!line.startsWith(DATA_PREFIX)) return;
  
  try {
    const data = JSON.parse(line.slice(DATA_PREFIX_LEN));
    const parts = data.response?.candidates?.[0]?.content?.parts;
    
    // 初始化状态标记（如果尚未存在）
    if (typeof state._thinkingActive === 'undefined') {
      state._thinkingActive = false;
    }

    if (parts) {
      for (const part of parts) {
        // ==========================================
        // 1. 处理思考内容 (Thought)
        // ==========================================
        if (part.thought === true) {
          // A. 处理签名 (透传逻辑)
          if (part.thoughtSignature) {
            state.reasoningSignature = part.thoughtSignature;
            if (state.sessionId && state.model) {
              setReasoningSignature(state.sessionId, state.model, part.thoughtSignature);
            }
          }

          // B. 通道一：发送原生 reasoning (供前端特定的思考框使用)
          callback({
            type: 'reasoning',
            reasoning_content: part.text || '',
            thoughtSignature: part.thoughtSignature || state.reasoningSignature || null
          });

          // C. 通道二：将思考内容注入到 Content 中 (实现双思维链)
          // 如果是新的一段思考开始，先发送 <thinking> 标签
          if (!state._thinkingActive) {
            callback({ type: 'text', content: '<thinking>\n' });
            state._thinkingActive = true;
          }
          // 同步发送思考文本到正文
          if (part.text) {
            callback({ type: 'text', content: part.text });
          }
        } 
        
        // ==========================================
        // 2. 处理普通文本 (Text)
        // ==========================================
        else if (part.text !== undefined) {
          // 如果之前处于思考状态，现在转为文本，说明思考结束，闭合标签
          if (state._thinkingActive) {
            callback({ type: 'text', content: '\n</thinking>\n\n' });
            state._thinkingActive = false;
          }
          
          callback({ type: 'text', content: part.text });
        } 
        
        // ==========================================
        // 3. 处理工具调用 (Function Call)
        // ==========================================
        else if (part.functionCall) {
          // 工具调用也会打断思考，需要闭合标签
          if (state._thinkingActive) {
            callback({ type: 'text', content: '\n</thinking>\n\n' });
            state._thinkingActive = false;
          }

          const toolCall = convertToToolCall(part.functionCall, state.sessionId, state.model);
          if (part.thoughtSignature) {
            toolCall.thoughtSignature = part.thoughtSignature;
            if (state.sessionId && state.model) {
              setToolSignature(state.sessionId, state.model, part.thoughtSignature);
            }
          }
          state.toolCalls.push(toolCall);
        }
      }
    }
    
    // ==========================================
    // 4. 处理流结束 (Finish)
    // ==========================================
    if (data.response?.candidates?.[0]?.finishReason) {
      // 如果流结束时标签仍未闭合，强制闭合
      if (state._thinkingActive) {
        callback({ type: 'text', content: '\n</thinking>\n\n' });
        state._thinkingActive = false;
      }

      if (state.toolCalls.length > 0) {
        callback({ type: 'tool_calls', tool_calls: state.toolCalls });
        state.toolCalls = [];
      }
      
      const usage = data.response?.usageMetadata;
      if (usage) {
        callback({
          type: 'usage',
          usage: {
            prompt_tokens: usage.promptTokenCount || 0,
            completion_tokens: usage.candidatesTokenCount || 0,
            total_tokens: usage.totalTokenCount || 0
          }
        });
      }
    }
  } catch {
    // 忽略 JSON 解析错误
  }
}

export {
  getLineBuffer,
  releaseLineBuffer,
  parseAndEmitStreamChunk,
  convertToToolCall,
  registerStreamMemoryCleanup,
  releaseToolCallObject
};
