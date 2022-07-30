import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
from param import args

Max_step = 3

# ==================================================================================================================== #
#                                                Graph reasoning                                                       #
# ==================================================================================================================== #
class Graphaggregation(nn.Module):
    def __init__(self):
        super(Graphaggregation, self).__init__()
        self.common_dim = 768
        self.w_g1 = nn.Linear(self.common_dim * 2, self.common_dim)
        self.w_g2 = nn.Linear(self.common_dim, self.common_dim)
        self.p_g = nn.Linear(self.common_dim, 1)
        # weight distribution
        self.softmax = nn.Softmax(dim=-2)
        # act func
        self.tanh = nn.Tanh()

    def forward(self, global_text, vertex_feats):
        global_feature = self.tanh(self.w_g1(vertex_feats) + self.w_g2(global_text))  # [256,37,1024]
        gate = self.softmax(self.p_g(global_feature))  # [256,37,1]
        final_feature = torch.matmul(gate.transpose(-1, -2), vertex_feats)  # [256,1,1024*2]
        return final_feature

class TextContext(nn.Module):
    def __init__(self):
        super(TextContext, self).__init__()
        self.common_dim = 768
        self.P = nn.Linear(self.common_dim, 1)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, textual_feats):
        # textual_feats [256, 26, 1024]
        z_h = self.P(textual_feats)   # [128, 20, 1]
        h = self.softmax(z_h)  # [128, 20, 1]
        c = torch.matmul(h.transpose(-1, -2), textual_feats)  #
        return c


class ReasoningInstructionExtraction(nn.Module):
    def __init__(self):
        super(ReasoningInstructionExtraction, self).__init__()
        self.common_dim = 768
        self.fc1 = nn.Linear(self.common_dim, self.common_dim)
        self.fc2 = nn.Linear(self.common_dim, self.common_dim)
        self.relu = nn.LeakyReLU()
        self.F = nn.Linear(self.common_dim, 1)  # d*1
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, embedding_output, instruction, history_vector):
        X_q = self.fc1(embedding_output)
        W_q = self.fc2(self.relu(X_q))
        a_q = self.softmax(self.F(W_q))
        s_g = torch.matmul(a_q.transpose(-1, -2), W_q) + history_vector + instruction
        return s_g

class HistoryVectorExtraction(nn.Module):
    def __init__(self):
        super(HistoryVectorExtraction, self).__init__()
        self.common_dim = 768
        self.fc1 = nn.Linear(self.common_dim * 2, self.common_dim)
        self.relu = nn.LeakyReLU()

        self.F = nn.Linear(self.common_dim , 1)  # d*1
        self.softmax = nn.Softmax(dim=-2)


    def forward(self, v_feats):
        W_v = self.fc1(v_feats)  # [256,37,1024]
        a_v = self.softmax(self.F(W_v))  # [256, 37, 1]
        s_v = torch.matmul(a_v.transpose(-1, -2), W_v)  # [256, 1, 1024]
        attention_map = a_v * W_v  # [256,37,1024 * 2]
        return s_v, attention_map

class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.common_dim = 768
        self.ins_extr = ReasoningInstructionExtraction()
        self.history_vector_predictor = HistoryVectorExtraction()
        self.steps = Max_step  # 3
        self.ins_layers = nn.ModuleList(
            [copy.deepcopy(self.ins_extr) for _ in range(self.steps)]
        )
        self.w1 = nn.Linear(self.common_dim * 2, self.common_dim)

        self.w2 = nn.Linear(self.common_dim * 2, self.common_dim)
        self.w3 = nn.Linear(self.common_dim, self.common_dim)

        # fc update vertex feat
        self.w4 = nn.Linear(self.common_dim * 2, self.common_dim)
        self.w5 = nn.Linear(self.common_dim, self.common_dim)
        self.w6 = nn.Sequential(
            nn.Linear(self.common_dim * 3, self.common_dim * 2),
            nn.ReLU(),
            nn.Linear(self.common_dim * 2, self.common_dim * 2),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, v_feats, embedding_output):
        # v_feats  [256, 37, 1024*2]
        # embedding_output [256, 26, 1024]

        history_vector_list = []
        attention_map_list = []
        batch_size = v_feats.size()[0]
        object_size = v_feats.size()[1]

        history_vector = torch.zeros(batch_size, 1, self.common_dim, device=v_feats.device)
        instruction = torch.zeros(batch_size, 1, self.common_dim, device=v_feats.device)
        for i in range(self.steps):
            instruction = self.ins_layers[i](embedding_output, instruction, history_vector)
            ins = instruction.repeat(1, object_size, 1)  # [256, 37, 1024]
            F = self.w1(v_feats)  # [256,37,1024]
            N = self.w2(v_feats) + self.w3(ins)  # [256,37,1024]
            adj = torch.matmul(F, N.transpose(-1, -2))  # [256,37,37]
            score = self.dropout(torch.softmax(adj, -1))   # [256,37,37]
            message = self.w4(v_feats) + self.w5(ins)  # [256,37,1024]
            MESSAGE = torch.matmul(score, message)  # [256,37,1024]
            v_feats = self.w6(torch.cat((v_feats, MESSAGE), dim=-1))  # [256,37,1024*2]
            history_vector, attention_map = self.history_vector_predictor(v_feats)
            history_vector_list.append(history_vector)
            attention_map_list.append(attention_map)
        return v_feats

class MRGT(nn.Module):
    def __init__(self):
        super(MRGT, self).__init__()
        self.v_hidden_size = 768
        self.hidden_size = 768
        self.common_dim = 768
        self.image_linear = nn.Linear(self.v_hidden_size, self.common_dim)
        self.text_linear = nn.Linear(self.hidden_size, self.common_dim)
        self.question_embedding = nn.Linear(self.hidden_size, self.common_dim)
        self.textual_context = TextContext()
        self.gnn = GNN()
        self.graph_attention_emb = Graphaggregation()

        # align
        self.fc1 = nn.Linear(self.common_dim * 2, self.common_dim)
        self.fc2 = nn.Linear(self.common_dim, self.common_dim)

    def forward(self, images, texts, embedding_output):
        # embedding_output: [256, 26, 768]
        v_feats = self.image_linear(images)  # [256, 37, 1024]

        t_feats = self.text_linear(texts)  # [256, 26, 1024]

        embedding_output = self.question_embedding(embedding_output)  # [256, 26, 1024]

        context = self.textual_context(t_feats)  # [256,1,1024]
        # context = self.textual_context(t_feats[:, 0])  # [256,1,1024]
        context_u = context.expand_as(v_feats).contiguous()  # [256,37,1024]

        vertex_feats = torch.cat((v_feats, context_u), dim=-1)  # [256, 37, 1024*2]
        # vertex_feats = t_feats  # [256, 37, 1024*2]

        update_feats = self.gnn(vertex_feats, embedding_output)  # [256, 37, 1024*2]

        qs = t_feats[:, 0].unsqueeze(1)  # [256, 1, 1024]
        q_s = qs.expand_as(v_feats).contiguous()  # [256, 37, 1024]

        graph_emb_v = self.graph_attention_emb(q_s, update_feats)  # torch.Size([256, 1, 1024*2])

        graph_emb_t = context  # [256, 1, 1024]

        # align
        graph_emb_v = self.fc1(graph_emb_v.squeeze(1))  # [256,1024]
        graph_emb_t = self.fc2(graph_emb_t.squeeze(1))  # [256,1024]

        final_output = graph_emb_v + graph_emb_t
        return final_output
