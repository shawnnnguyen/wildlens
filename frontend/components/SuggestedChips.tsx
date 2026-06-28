import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { Fonts } from '../constants/theme';

interface Props {
  chips: string[];
  onPress: (chip: string) => void;
}

export default function SuggestedChips({ chips, onPress }: Props) {
  return (
    <View style={styles.wrapper}>
      <Text style={styles.label}>SUGGESTED</Text>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.row}>
        {chips.map(chip => (
          <TouchableOpacity key={chip} style={styles.chip} onPress={() => onPress(chip)}>
            <Text style={styles.chipText}>{chip}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: { gap: 8 },
  label: { fontFamily: 'SpaceMono_400Regular', fontSize: 8.5, letterSpacing: 1.6, color: '#8a7350' },
  row: { flexDirection: 'row', gap: 8 },
  chip: {
    borderWidth: 1, borderColor: 'rgba(194,129,74,0.55)',
    backgroundColor: 'rgba(194,129,74,0.08)',
    paddingHorizontal: 13, paddingVertical: 8, borderRadius: 20,
  },
  chipText: { fontFamily: Fonts.body, fontSize: 14.5, color: '#3a2f1d' },
});
