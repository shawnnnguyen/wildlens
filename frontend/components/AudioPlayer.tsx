import { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Colors, Fonts } from '../constants/theme';

const BARS = [6,10,16,9,20,14,8,18,12,22,16,10,7,15,19,11,21,13,8,17,14,9,18,12,6,16,20,10,8,14,11,17,9,13,7];

interface Props { guideName: string }

export default function AudioPlayer({ guideName }: Props) {
  const [playing, setPlaying] = useState(false);

  return (
    <View style={styles.card}>
      <View style={styles.row}>
        <TouchableOpacity style={styles.playBtn} onPress={() => setPlaying(p => !p)}>
          <Text style={styles.playGlyph}>{playing ? '❚❚' : '▶'}</Text>
        </TouchableOpacity>
        <View style={styles.right}>
          <View style={styles.titleRow}>
            <Text style={styles.title}>Narrated by {guideName}</Text>
            <Text style={styles.duration}>0:48</Text>
          </View>
          <View style={styles.waveform}>
            {BARS.map((h, i) => (
              <View key={i} style={[styles.bar, { height: h }]} />
            ))}
          </View>
        </View>
      </View>
      <Text style={styles.voiceLabel}>VOICE · WARM · UNHURRIED</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: { backgroundColor: Colors.dark, borderRadius: 10, padding: 16 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 13 },
  playBtn: {
    width: 46, height: 46, borderRadius: 23,
    backgroundColor: Colors.amber,
    alignItems: 'center', justifyContent: 'center',
  },
  playGlyph: { color: Colors.dark, fontSize: 14 },
  right: { flex: 1 },
  titleRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 9 },
  title: { fontFamily: Fonts.display, fontSize: 17, color: Colors.cream },
  duration: { fontFamily: Fonts.mono, fontSize: 9, color: 'rgba(243,236,222,0.55)' },
  waveform: { flexDirection: 'row', alignItems: 'center', gap: 2, height: 24 },
  bar: { width: 3, backgroundColor: 'rgba(243,236,222,0.45)', borderRadius: 2 },
  voiceLabel: {
    fontFamily: Fonts.mono, fontSize: 8.5,
    letterSpacing: 1.2, color: 'rgba(243,236,222,0.45)',
    marginTop: 13,
  },
});
