import os
import requests
import json
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd

from backtest_logic import buscar_dados_api, detectar_sinais, executar_simulacao, calcular_metricas

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'SUA_CHAVE_AQUI_PARA_TESTE_LOCAL')

def create_analysis_prompt(metricas, trades):
    """Cria o prompt para a API Gemini."""
    trades_str = ""
    for trade in trades[:10]:
        trades_str += f"- Trade em {pd.to_datetime(trade['data_entrada']).strftime('%d/%m/%Y')}: {trade['resultado']}, P&L: $ {float(trade['pnl_financeiro']):.2f}\n"
    prompt = f"""
    Você é um analista de risco e estrategista de trading profissional. Analise os resultados de um backtest da estratégia "Invert 50".

    **Resultados do Backtest:**
    - **Métricas Gerais:** {json.dumps(metricas, indent=2, ensure_ascii=False)}
    - **Amostra de Trades:**
    {trades_str}

    **Sua Análise:**
    Com base nos dados, forneça uma análise em 3 partes, usando Markdown:
    1.  **### Diagnóstico Geral:** Resuma o desempenho. Foi lucrativo? Consistente?
    2.  **### Pontos Fortes:** Identifique os aspectos positivos.
    3.  **### Pontos de Melhoria e Riscos:** Aponte as fraquezas e sugira otimizações.
    Seja direto, objetivo e use uma linguagem que um trader entenderia.
    """
    return prompt

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/backtest', methods=['POST'])
def handle_backtest_request():
    """Executa o backtest e retorna os resultados completos."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Pedido inválido."}), 400

        ativo = data.get('ativo')
        data_inicio = data.get('data_inicio')
        data_fim = data.get('data_fim')
        lotes = int(data.get('lotes', 1))
        valor_ponto = float(data.get('valor_ponto', 1.0))
        estrategia = data.get('estrategia', 'venda')

        if not all([ativo, data_inicio, data_fim]):
            return jsonify({"error": "Parâmetros 'ativo', 'data_inicio' e 'data_fim' são obrigatórios."}), 400

        dados_historicos = buscar_dados_api(ativo, data_inicio, data_fim)
        if dados_historicos is None or dados_historicos.empty:
            return jsonify({"error": "Não foi possível obter dados históricos com yfinance. Verifique o ticker do ativo."}), 404

        sinais_df = detectar_sinais(dados_historicos, estrategia)
        if sinais_df.empty:
            return jsonify({"error": "Nenhum trade foi gerado pela estratégia neste período."}), 200

        resultados_df = executar_simulacao(dados_historicos, sinais_df, tipo_operacao=estrategia)
        metricas = calcular_metricas(resultados_df, lotes, valor_ponto)

        historico_formatado = [{"time": int(index.timestamp()), "open": row['open'], "high": row['high'], "low": row['low'], "close": row['close']} for index, row in dados_historicos.iterrows()]
        
        resultados_df['data_entrada'] = pd.to_datetime(resultados_df['data_entrada']).astype(str)
        if 'data_saida' in resultados_df.columns:
            resultados_df['data_saida'] = pd.to_datetime(resultados_df['data_saida']).astype(str)
        
        trades_detalhados = resultados_df.to_dict(orient='records')

        return jsonify({
            "metricas": metricas,
            "trades": trades_detalhados,
            "historico_ohlc": historico_formatado
        })

    except Exception as e:
        print(f"ERRO CRÍTICO NO ENDPOINT /backtest: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Ocorreu um erro crítico no servidor."}), 500

@app.route('/analyze-results', methods=['POST'])
def analyze_results_with_gemini():
    """Chama a API Gemini para analisar os resultados."""
    try:
        backtest_data = request.get_json()
        prompt = create_analysis_prompt(backtest_data.get('metricas'), backtest_data.get('trades'))
        
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        response = requests.post(gemini_url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        analysis_text = result['candidates'][0]['content']['parts'][0]['text']
        return jsonify({"analysis": analysis_text})

    except Exception as e:
        print(f"ERRO CRÍTICO NO ENDPOINT /analyze-results: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Ocorreu um erro interno na análise da IA."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
